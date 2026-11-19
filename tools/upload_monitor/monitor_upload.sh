#!/bin/bash
# 便捷监控脚本

# 设置工作目录为main目录
MAIN_DIR="/research/cbim/vast/sf895/code/wsg/main"
cd "$MAIN_DIR"

echo "=== HuggingFace 上传监控面板 ==="
echo "时间: $(date)"
echo "工作目录: $MAIN_DIR"
echo ""

# 检查进程
echo "1. 运行进程:"
ps aux | grep -E "pipeline31|huggingface" | grep -v grep | grep -v monitor || echo "   未找到上传进程"
echo ""

# 显示最新状态
echo "2. 系统状态 (最新5条):"
tail -5 upload_status.log 2>/dev/null || echo "   暂无状态日志"
echo ""

# 显示上传进度
echo "3. 上传进度:"
if [ -f upload_progress.json ]; then
    python3 -c "
import json
with open('upload_progress.json', 'r') as f:
    progress = json.load(f)
    for key, value in progress.items():
        status = '✓ 完成' if value.get('completed') else '✗ 进行中/失败'
        print(f\"   {value['folder']}: {status}\")"
else
    echo "   暂无进度记录"
fi
echo ""

# 显示日志文件
echo "4. 日志文件:"
ls -lah *.log 2>/dev/null | grep -v "^total" || echo "   暂无日志文件"
echo ""

echo "提示: "
echo "- 查看实时进度: tail -f upload_detail.log"
echo "- 持续监控: watch -n 10 bash $0"