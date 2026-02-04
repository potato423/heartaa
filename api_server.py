"""
HeartMuLa API Server
基于官方 heartlib 库的 FastAPI 服务

参考文档：
- GitHub: https://github.com/HeartMuLa/heartlib
- HuggingFace: https://hf-mirror.com/HeartMuLa/HeartMuLa-oss-3B

部署方式：
- Docker 镜像只包含代码和依赖
- 模型通过共绩算力存储卷挂载到 /data/ckpt
- 首次启动自动下载模型到存储卷

API 端点：
- POST /generate/music  - 提交音乐生成任务
- GET  /jobs/{job_id}   - 查询任务状态
- GET  /download_track/{job_id} - 下载生成的音频
- DELETE /jobs/{job_id} - 删除任务
"""

import os
import gc
import uuid
import asyncio
import subprocess
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============== 配置 ==============
# 模型路径 - 挂载存储卷的目录
MODEL_PATH = os.environ.get("MODEL_PATH", "/data/ckpt")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/heartmula_outputs")
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "2"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "3B")

# GPU 设备配置
MULA_DEVICE = os.environ.get("MULA_DEVICE", "cuda:0")
CODEC_DEVICE = os.environ.get("CODEC_DEVICE", "cuda:0")

# 是否使用懒加载（单GPU内存不足时开启）
LAZY_LOAD = os.environ.get("LAZY_LOAD", "false").lower() == "true"

# HuggingFace 镜像（国内加速）
HF_MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# ============== 任务存储 ==============
jobs: Dict[str, Dict[str, Any]] = {}

# 线程池用于执行阻塞的模型推理
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS)

# 模型加载状态
model_status = {
    "loaded": False,
    "loading": False,
    "error": None
}

# ============== Pydantic 模型 ==============
class GenerateMusicRequest(BaseModel):
    """音乐生成请求参数 - 严格遵循 heartlib 官方参数"""
    prompt: str = Field(..., description="生成提示词/描述")
    lyrics: str = Field(default="[instrumental]", description="歌词内容，格式参考官方文档")
    tags: Optional[str] = Field(default=None, description="风格标签，逗号分隔如: piano,happy,romantic")
    duration_ms: int = Field(default=60000, ge=1000, le=240000, description="音频时长(毫秒)，最大240000")
    
    # 高级参数 - 来自官方 run_music_generation.py
    topk: int = Field(default=50, ge=1, le=500, description="Top-k 采样参数")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="采样温度")
    cfg_scale: float = Field(default=1.5, ge=1.0, le=5.0, description="Classifier-free guidance scale")
    seed: Optional[int] = Field(default=None, description="随机种子，用于复现")


class JobStatus(BaseModel):
    """任务状态响应"""
    id: str
    status: str  # queued, processing, completed, failed
    error_msg: Optional[str] = None
    audio_path: Optional[str] = None
    generation_time_seconds: Optional[float] = None
    prompt: Optional[str] = None
    lyrics: Optional[str] = None
    tags: Optional[str] = None
    duration_ms: Optional[int] = None
    created_at: Optional[str] = None


# ============== FastAPI 应用 ==============
app = FastAPI(
    title="HeartMuLa API",
    description="基于 HeartMuLa 开源模型的音乐生成 API 服务",
    version="1.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== 模型下载与加载 ==============

def check_models_exist() -> bool:
    """检查模型文件是否已存在于存储卷"""
    required_paths = [
        Path(MODEL_PATH) / "gen_config.json",
        Path(MODEL_PATH) / "tokenizer.json",
        Path(MODEL_PATH) / "HeartMuLa-oss-3B",
        Path(MODEL_PATH) / "HeartCodec-oss",
    ]
    
    for p in required_paths:
        if not p.exists():
            print(f"[HeartMuLa] 缺少文件: {p}")
            return False
    
    print(f"[HeartMuLa] 模型文件已存在于存储卷: {MODEL_PATH}")
    return True


def download_models():
    """
    下载模型到存储卷
    使用 HuggingFace 国内镜像加速
    """
    print(f"[HeartMuLa] 开始下载模型到存储卷: {MODEL_PATH}")
    print(f"[HeartMuLa] 使用镜像: {HF_MIRROR}")
    
    # 设置 HuggingFace 镜像
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    
    try:
        from huggingface_hub import snapshot_download
        
        model_path = Path(MODEL_PATH)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # 下载列表
        downloads = [
            ("HeartMuLa/HeartMuLaGen", str(model_path)),
            ("HeartMuLa/HeartMuLa-RL-oss-3B-20260123", str(model_path / "HeartMuLa-oss-3B")),
            ("HeartMuLa/HeartCodec-oss-20260123", str(model_path / "HeartCodec-oss")),
        ]
        
        for repo_id, local_dir in downloads:
            print(f"[HeartMuLa] 下载 {repo_id} -> {local_dir}")
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"[HeartMuLa] ✓ {repo_id} 下载完成")
        
        print(f"[HeartMuLa] 所有模型下载完成!")
        return True
        
    except Exception as e:
        print(f"[HeartMuLa] 模型下载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============== 全局模型实例（延迟加载）==============
_model_instance = None
_model_lock = asyncio.Lock()


def cleanup_gpu_memory(pipeline=None):
    """
    清理 GPU 显存，防止 OOM
    参考 HeartMuLa-Studio 的实现
    """
    import torch
    try:
        # 1. 如果有 pipeline，先清理 KV cache
        if pipeline is not None:
            try:
                if hasattr(pipeline, 'mula') and hasattr(pipeline.mula, 'reset_caches'):
                    pipeline.mula.reset_caches()
                    print("[HeartMuLa] KV cache 已重置")
            except Exception as e:
                print(f"[HeartMuLa] 重置 KV cache 时出错: {e}")
        
        # 2. Python 垃圾回收
        gc.collect()
        
        # 3. CUDA 显存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # 打印当前显存状态
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            free = torch.cuda.mem_get_info()[0] / 1024**3
            print(f"[HeartMuLa] 显存状态: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB, 可用 {free:.2f}GB")
        
        print("[HeartMuLa] GPU 显存清理完成")
    except Exception as e:
        print(f"[HeartMuLa] 清理显存时出错: {e}")


def get_model():
    """获取或初始化模型实例"""
    global _model_instance, model_status
    
    if _model_instance is None:
        model_status["loading"] = True
        
        try:
            print(f"[HeartMuLa] 正在加载模型...")
            print(f"[HeartMuLa] MODEL_PATH: {MODEL_PATH}")
            print(f"[HeartMuLa] VERSION: {MODEL_VERSION}")
            print(f"[HeartMuLa] MULA_DEVICE: {MULA_DEVICE}")
            print(f"[HeartMuLa] CODEC_DEVICE: {CODEC_DEVICE}")
            print(f"[HeartMuLa] LAZY_LOAD: {LAZY_LOAD}")
            
            # 导入 heartlib（延迟导入，避免启动时就加载大模型）
            import torch
            from heartlib import HeartMuLaGenPipeline
            from heartlib.heartcodec.modeling_heartcodec import HeartCodec
            
            # ============ 关键修复：Monkey patch HeartCodec.detokenize ============
            # 问题：HeartMuLa 生成的 token 有时会超出 codec 词汇表范围（8192）
            # 比如 EOS token（8193），这会导致 "index out of bounds" 错误
            # 解决：在 detokenize 前将 frames 限制在有效范围 [0, 8191]
            # 参考：https://github.com/fspecii/HeartMuLa-Studio
            
            _original_detokenize = HeartCodec.detokenize
            
            @torch.inference_mode()
            def _patched_detokenize(self, frames, *args, **kwargs):
                # Clamp frame indices to valid codec vocabulary range [0, 8191]
                # The model may generate special tokens (EOS=8193, etc.) that are out of bounds
                CODEC_VOCAB_SIZE = 8192  # From codec_config.json codebook_size
                frames_clamped = frames.clamp(0, CODEC_VOCAB_SIZE - 1)
                print(f"[HeartMuLa] Frames clamped to [0, {CODEC_VOCAB_SIZE - 1}]")
                return _original_detokenize(self, frames_clamped, *args, **kwargs)
            
            HeartCodec.detokenize = _patched_detokenize
            print(f"[HeartMuLa] HeartCodec.detokenize 已打补丁，修复 index out of bounds 错误")
            # ============ 补丁结束 ============
            
            # ============ 关键修复 2：Sequential Offload ============
            # 问题：HeartMuLa (3B) 占用 ~23GB 显存，codec 解码时没有足够空间
            # 解决：在 postprocess (codec 解码) 前，先把 HeartMuLa 移到 CPU
            # 参考：https://github.com/fspecii/HeartMuLa-Studio
            
            from heartlib.pipelines.music_generation import HeartMuLaGenPipeline as _Pipeline
            _original_postprocess = _Pipeline.postprocess
            
            def _patched_postprocess(self, model_outputs, **kwargs):
                """在 codec 解码前，释放 HeartMuLa 显存"""
                import gc as _gc  # 在函数内部导入，确保可见
                
                print(f"[HeartMuLa] Sequential Offload: 释放 HeartMuLa 显存...")
                
                # 1. 重置 KV cache
                if hasattr(self, 'mula') and hasattr(self.mula, 'reset_caches'):
                    self.mula.reset_caches()
                    print(f"[HeartMuLa] KV cache 已重置")
                
                # 2. 将 HeartMuLa 移到 CPU
                if hasattr(self, '_mula') and self._mula is not None:
                    self._mula.to("cpu")
                    print(f"[HeartMuLa] HeartMuLa 已移至 CPU")
                
                # 3. 清理 GPU 显存
                _gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # 打印显存状态
                allocated = torch.cuda.memory_allocated() / 1024**3
                free = torch.cuda.mem_get_info()[0] / 1024**3
                print(f"[HeartMuLa] 显存释放后: 已分配 {allocated:.2f}GB, 可用 {free:.2f}GB")
                
                # 4. 确保 HeartCodec 在 GPU 上进行解码
                if hasattr(self, '_codec') and self._codec is not None:
                    codec_device = next(self._codec.parameters()).device
                    if codec_device.type == "cpu":
                        self._codec.to(torch.device(CODEC_DEVICE))
                        print(f"[HeartMuLa] HeartCodec 已移至 GPU 进行解码")
                
                # 5. 调用原始 postprocess（codec 解码）
                result = _original_postprocess(self, model_outputs, **kwargs)
                
                # 5. 解码完成后，不立即移回 GPU（会 OOM）
                # HeartMuLa 留在 CPU，下次生成时再移回
                print(f"[HeartMuLa] Codec 解码完成，HeartMuLa 保持在 CPU")
                
                return result
            
            _Pipeline.postprocess = _patched_postprocess
            print(f"[HeartMuLa] Pipeline.postprocess 已打补丁，实现 Sequential Offload")
            # ============ 补丁 2 结束 ============
            
            # 设置设备
            device = {
                "mula": torch.device(MULA_DEVICE),
                "codec": torch.device(CODEC_DEVICE)
            }
            
            # 设置数据类型
            dtype = {
                "mula": torch.bfloat16,
                "codec": torch.float32
            }
            
            _model_instance = HeartMuLaGenPipeline.from_pretrained(
                pretrained_path=MODEL_PATH,
                device=device,
                dtype=dtype,
                version=MODEL_VERSION,
                lazy_load=LAZY_LOAD
            )
            
            model_status["loaded"] = True
            model_status["loading"] = False
            print(f"[HeartMuLa] 模型加载完成!")
            
        except Exception as e:
            model_status["loading"] = False
            model_status["error"] = str(e)
            import traceback
            traceback.print_exc()
            raise e
    
    return _model_instance


def generate_music_sync(job_id: str, request: GenerateMusicRequest):
    """
    同步执行音乐生成（在线程池中运行）
    严格遵循 heartlib 官方 API（HeartMuLaGenPipeline）
    包含 OOM 防护：生成前后清理显存
    """
    pipeline = None
    lyrics_file = None
    tags_file = None
    
    try:
        jobs[job_id]["status"] = "processing"
        start_time = datetime.now()
        
        print(f"[HeartMuLa] Job {job_id} 开始生成...")
        print(f"[HeartMuLa] Prompt: {request.prompt}")
        print(f"[HeartMuLa] Lyrics: {request.lyrics[:100]}...")
        print(f"[HeartMuLa] Tags: {request.tags}")
        print(f"[HeartMuLa] Duration: {request.duration_ms}ms")
        
        # ============ 生成前清理显存 ============
        # 防止显存碎片化导致 OOM
        cleanup_gpu_memory()
        
        # 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp3")
        
        # 获取模型 pipeline
        pipeline = get_model()
        
        # ============ 确保 HeartMuLa 在 GPU 上 ============
        # 上次生成后 HeartMuLa 可能留在 CPU，需要移回 GPU
        import torch
        if hasattr(pipeline, '_mula') and pipeline._mula is not None:
            mula_device = next(pipeline._mula.parameters()).device
            target_device = torch.device(MULA_DEVICE)
            if mula_device != target_device:
                print(f"[HeartMuLa] HeartMuLa 在 {mula_device}，需要移到 {target_device}")
                # 先把 HeartCodec 移到 CPU 腾出空间
                if hasattr(pipeline, '_codec') and pipeline._codec is not None:
                    pipeline._codec.to("cpu")
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f"[HeartMuLa] HeartCodec 已移至 CPU")
                # 再把 HeartMuLa 移回 GPU
                pipeline._mula.to(target_device)
                print(f"[HeartMuLa] HeartMuLa 已移至 {target_device}")
        
        # 准备输入 - HeartMuLaGenPipeline 接受字符串或文件路径
        # 如果是纯文本，直接传递；如果需要文件，创建临时文件
        lyrics_content = request.lyrics.lower()  # 必须小写
        
        # tags 必须是小写、逗号分隔、无空格的格式
        # 例如: "piano,happy,romantic" 而不是 "piano, happy, romantic"
        tags_content = request.tags or "instrumental"
        tags_content = tags_content.lower().replace(" ", "")  # 小写并移除空格
        
        # HeartMuLaGenPipeline 可以接受字符串或文件路径
        # 为了简单起见，我们创建临时文件
        lyrics_file = os.path.join(OUTPUT_DIR, f"{job_id}_lyrics.txt")
        with open(lyrics_file, "w", encoding="utf-8") as f:
            f.write(lyrics_content)
        
        tags_file = os.path.join(OUTPUT_DIR, f"{job_id}_tags.txt")
        with open(tags_file, "w", encoding="utf-8") as f:
            f.write(tags_content)
        
        # 调用 HeartMuLaGenPipeline
        # pipeline(inputs, **kwargs)
        inputs = {
            "lyrics": lyrics_file,
            "tags": tags_file,
        }
        
        pipeline(
            inputs,
            save_path=output_path,
            max_audio_length_ms=request.duration_ms,
            topk=request.topk,
            temperature=request.temperature,
            cfg_scale=request.cfg_scale,
        )
        
        # 计算生成时间
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        # 更新任务状态
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["audio_path"] = output_path
        jobs[job_id]["generation_time_seconds"] = generation_time
        
        print(f"[HeartMuLa] Job {job_id} 完成! 耗时: {generation_time:.2f}s")
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"[HeartMuLa] Job {job_id} 失败: {error_msg}")
        traceback.print_exc()
        
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error_msg"] = error_msg
    
    finally:
        # ============ 生成后清理（无论成功或失败）============
        # 清理临时文件
        if lyrics_file and os.path.exists(lyrics_file):
            try:
                os.remove(lyrics_file)
            except Exception:
                pass
        if tags_file and os.path.exists(tags_file):
            try:
                os.remove(tags_file)
            except Exception:
                pass
        
        # 清理 GPU 显存（包括 KV cache）
        # 这是防止连续生成时 OOM 的关键
        cleanup_gpu_memory(pipeline)


# ============== API 端点 ==============

@app.get("/")
async def root():
    """健康检查"""
    return {
        "service": "HeartMuLa API",
        "status": "running",
        "model_path": MODEL_PATH,
        "version": MODEL_VERSION,
        "model_status": model_status
    }


@app.get("/health")
async def health():
    """健康检查端点"""
    return {"status": "healthy", "model_status": model_status}


@app.get("/model/status")
async def get_model_status():
    """获取模型状态"""
    models_exist = check_models_exist()
    return {
        "models_downloaded": models_exist,
        "model_loaded": model_status["loaded"],
        "model_loading": model_status["loading"],
        "error": model_status["error"],
        "model_path": MODEL_PATH
    }


@app.post("/model/download")
async def trigger_model_download(background_tasks: BackgroundTasks):
    """
    手动触发模型下载
    用于首次部署时下载模型到存储卷
    """
    if check_models_exist():
        return {"status": "already_exists", "message": "模型已存在于存储卷"}
    
    # 在后台下载
    background_tasks.add_task(download_models)
    return {"status": "downloading", "message": "模型下载已开始，请稍后查询 /model/status"}


@app.post("/generate/music")
async def generate_music(request: GenerateMusicRequest, background_tasks: BackgroundTasks):
    """
    提交音乐生成任务
    
    返回 job_id，通过 /jobs/{job_id} 查询状态
    """
    # 检查模型是否就绪
    if not check_models_exist():
        raise HTTPException(
            status_code=503, 
            detail="模型未下载，请先调用 POST /model/download 或等待自动下载完成"
        )
    
    job_id = str(uuid.uuid4())
    
    # 初始化任务状态
    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "error_msg": None,
        "audio_path": None,
        "generation_time_seconds": None,
        "prompt": request.prompt,
        "lyrics": request.lyrics,
        "tags": request.tags,
        "duration_ms": request.duration_ms,
        "created_at": datetime.now().isoformat()
    }
    
    # 在后台线程池中执行生成任务
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, generate_music_sync, job_id, request)
    
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """查询任务状态"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]


@app.get("/download_track/{job_id}")
async def download_track(job_id: str):
    """下载生成的音频文件"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Audio not ready. Status: {job['status']}"
        )
    
    audio_path = job.get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        audio_path,
        media_type="audio/mpeg",
        filename=f"{job_id}.mp3"
    )


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """删除任务及其生成的文件"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # 删除音频文件
    audio_path = job.get("audio_path")
    if audio_path and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except Exception as e:
            print(f"[HeartMuLa] 删除文件失败: {e}")
    
    # 从任务列表中移除
    del jobs[job_id]
    
    return {"status": "deleted", "job_id": job_id}


@app.get("/jobs")
async def list_jobs(limit: int = 50):
    """列出所有任务（调试用）"""
    job_list = list(jobs.values())[-limit:]
    return {"jobs": job_list, "total": len(jobs)}


# ============== 启动事件 ==============

@app.on_event("startup")
async def startup_event():
    """服务启动时的初始化 - 自动下载模型"""
    import torch
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[HeartMuLa] API 服务启动")
    print(f"[HeartMuLa] 输出目录: {OUTPUT_DIR}")
    print(f"[HeartMuLa] 模型路径: {MODEL_PATH}")
    print(f"[HeartMuLa] 模型版本: {MODEL_VERSION}")
    print(f"[HeartMuLa] HF 国内镜像: {HF_MIRROR}")
    print(f"[HeartMuLa] 最大并发: {MAX_CONCURRENT_JOBS}")
    
    # ============ GPU 显存优化配置 ============
    # 设置 PyTorch 内存分配器配置，减少显存碎片化
    # 参考: https://pytorch.org/docs/stable/notes/cuda.html#environment-variables
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        print(f"[HeartMuLa] 已设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    
    # 显示 GPU 信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[HeartMuLa] GPU: {gpu_name} ({gpu_mem:.1f}GB)")
        
        # 显存使用建议
        if gpu_mem < 16:
            print(f"[HeartMuLa] ⚠️ 显存 < 16GB，建议使用短时长（duration_ms <= 30000）")
        elif gpu_mem < 24:
            print(f"[HeartMuLa] 💡 显存 < 24GB，建议使用中等时长（duration_ms <= 60000）")
        else:
            print(f"[HeartMuLa] ✓ 显存充足，可生成较长音频")
    else:
        print(f"[HeartMuLa] ⚠️ 未检测到 CUDA GPU，将使用 CPU（非常慢）")
    
    # 检查模型是否已存在，不存在则自动下载
    if check_models_exist():
        print(f"[HeartMuLa] 模型已存在于存储卷，服务就绪")
    else:
        print(f"[HeartMuLa] 模型不存在，开始自动下载...")
        print(f"[HeartMuLa] 使用国内镜像: {HF_MIRROR}")
        # 在后台线程下载，不阻塞服务启动
        loop = asyncio.get_event_loop()
        loop.run_in_executor(executor, download_models)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
