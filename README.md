# 悬浮提词器

手机网页提词器：在网页里用前置摄像头录像，说话时台词往下滚、停下就停（听的是录像自己的麦克风，不用苹果语音识别，所以不会抢走录像的声音）；可以横拿手机拍竖版。带美白、瘦脸（只动脸）。

网址：https://taozihenzhuai666.github.io/tici/

## 第三方

美白、瘦脸找脸用 Google MediaPipe（@mediapipe/tasks-vision 0.10.14 人脸检测 blaze_face_short_range），Apache-2.0 许可。
模型程序（9.4MB）优先从国内镜像 registry.npmmirror.com 下载，不通再用本站 `mp/wasm/` 和 jsdelivr；下过一次存在手机里。
找脸在后台线程 `face-worker.js` 里跑，录像不掉帧；后台线程起不来就退回页面里跑。
`mp/` 里是同版本文件的备份（只留 iPhone 用的 SIMD 版；老设备要的 nosimd 版从镜像下）。
