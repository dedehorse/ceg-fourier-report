CEG 和弦的傅里叶变换与采样率分析 —— 提交说明

1. 先打开 index.html（网页版报告）
   - 可直接播放 8 段 5.000 s 音频，查看全部 9 张图与数据表。
   - 页面离线可用，无需联网；请保持 submission 文件夹结构不变。

2. 正式报告
   - CEG和弦傅里叶分析报告.pdf  （14 页，WPS 导出，可直接提交/打印）
   - CEG和弦傅里叶分析报告.docx （可编辑，图片已嵌入）

3. 音频（audio/，均为 16 bit 单声道 WAV，时长 5.000 s）
   01_original_44100.wav        原始信号（峰值归一化到 0.850）
   02_22050_resampled.wav       抗混叠降到 22.05 kHz
   03_8000_resampled.wav        抗混叠降到 8 kHz
   04_1000_naive.wav            直接抽取到 1 kHz（暴露混叠）
   05_512_naive.wav             直接抽取到 512 Hz（严重混叠）
   06_512_antialias.wav         抗混叠降到 512 Hz（对照）
   07_512_to_1024_fir.wav       512 Hz → 1024 Hz，零插值 + FIR
   08_512_to_1024_linear.wav    512 Hz → 1024 Hz，线性内插

4. 图像与数据
   - figures/fig01–fig09：报告中的 9 张图（220 dpi PNG）
   - analysis_summary.json：全部实验指标（机器可读）
   - CEG.wav：原始 8.000 s 文件

5. 复现
   - 环境：Python 3.12，numpy 1.26、scipy 1.13、matplotlib 3.8
   - 在本目录执行：python analyse_ceg.py
   - 该脚本会重新生成 figures/、audio/ 与 analysis_summary.json。

注：封面姓名、学号、专业班级留空，请提交前填写。
