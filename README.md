# Qwen3-TTS-GGUF minimal

Qwen3-TTS-0.6B 的 GGUF 推理引擎极简封装：一条命令把文本合成成 wav，
支持内置音色与声音克隆（参考音频 + 台词转写）。

基于 [HaujetZhao/Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF)
（MIT License，Copyright (c) 2026 HaujetZhao）裁剪而来，只保留运行时
`qwen3_tts_gguf/` + 一个命令行入口。模型来自
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)（Apache-2.0）。

## 快速开始

```bash
pip install -r requirements.txt

# 内置音色
python render.py --model-dir <模型包目录> --text "今天天气真不错。" --speaker Vivian

# 声音克隆（参考音频原话转写必须逐字覆盖参考音频）
python render.py --model-dir <模型包目录> --text "今天天气真不错。" \
    --voice-wav ref.wav --ref-text "参考音频里那个人说的原话"
```

内置音色：`Vivian` `Serena` `Ono_Anna` `Sohee` `Ryan` `Aiden`
`Uncle_Fu` `Eric` `Dylan`（也可用说话人 ID 2800-3071）。
`--instruct` 支持自然语言渲染指令（如"用温柔的语气说"）。
`--language` 支持 chinese/english/japanese/korean 等语种与方言。

## 模型包

需要 Qwen3-TTS-0.6B 的 GGUF 转换套件目录（talker/predictor gguf、
decoder/codec_encoder/speaker_encoder onnx、`embeddings/`、`tokenizer.json`）。
转换方法见原上游仓库的 1x/2x 号导出脚本；量化说明与实验记录随模型包
发布（ModelScope）。

推荐直接用我们发布的量化模型包
[Qwen3-TTS-0.6B-GGUF-Lunavox-int8emb](https://modelscope.cn/models/ArchitectLi/Qwen3-TTS-0.6B-GGUF-Lunavox-int8emb)
（含语音克隆双编码器 codec_encoder/speaker_encoder，量化说明与
实测结论见模型卡）。

## 声音克隆须知（实测结论）

1. **参考音频必须是一段完整说完的话**——断在半句/半音素的录音必失败
   （克隆输出失控到步数上限）。台词转写要逐字覆盖喂进去的音频。
2. **参考音频时长官方建议 3-10 秒**，只掐头尾静音，不要截断正文。
3. **极端嗓音克隆会失效**——极低/极高的音域模型学不像（实测 f0 漂移
   80%+ 且生成失稳），效果预期要放低。

克隆是"示范学习"：参考台词 + 参考音频配成一个示范，文本与音频对不上
示范就是坏的，生成会失控。台词可以用户填写、ASR 转写，或用
`--zero-shot` 走免台词弱克隆（只用声纹向量，稳定但相似度上限低）。

## 注意事项

- **talker 固定文件名**：引擎按 `qwen3_tts_talker.q5_k.gguf` 读取
  talker（文件名约定来自上游 HaujetZhao/Qwen3-TTS-GGUF 的
  `engine.py` 路径定义，本仓库未改动此处）。文件名是约定，量化
  格式不是——GGUF 自描述，内容为 q4_0 等其他档位照常加载，用
  软链对齐文件名即可，不影响效果。套件里若是其他档位文件名，
  `render.py` 会自动在模型包旁建 `<包名>_vdir` 别名目录（软链/
  硬链，不复制大文件）；手动调引擎 API 时同样处理。
- **模型包组件请勿跨包混用**：不同底模变体的训练产物不同，
  跨包混用组件无法正常工作。
- 编码器保持上游 fp16 原版：int8 量化的 codes 一致率仅 10%，等于换内容。

## 与上游的差异（本仓库改动清单）

- `qwen3_tts_gguf/inference/assets.py`：embeddings 加载支持 int8 表 +
  scale，fp16 存储的 codec 表加载时上采 float32。
- `qwen3_tts_gguf/inference/encoder.py`：codec_encoder 输入按 960 样本
  对齐补零——上游 ONNX 图对任意长度音频会触发 Reshape 错误
  （图内 ceil/floor 差一帧），引擎自产音频不触发，真实录音会触发。

## License

MIT（见 LICENSE）。保留上游版权与许可声明。
