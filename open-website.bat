@echo off
echo Starting Joseph Bidias Portfolio Website...
cd "C:\Users\josep\Desktop\Quant Researcher AI_ML Specialist\Joseph-Bidias-Quant-AI-ML-Portfolio\docs"
start python -m http.server 8083
timeout /t 3 /nobreak
start http://localhost:8083
echo Website should open in your browser at http://localhost:8083
pause