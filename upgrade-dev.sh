#!/bin/bash
# Full upgrade: stop server -> build exllamav3 dev at full speed -> retune host
# RAM setting -> restart -> verify. Server down for the duration by design:
# the build wants the RAM the server is holding.
set -u
EX=/home/dushyant30suthar/Projects/exllamav3
TB=/home/dushyant30suthar/Projects/tabbyAPI
ST=/home/dushyant30suthar/.local/state/opencode/providers/exl3

echo "=== 1. stopping server ==="
pkill -f "main[.]py --config" 2>/dev/null
for i in $(seq 1 30); do pgrep -f "main[.]py --config" >/dev/null || break; sleep 2; done
pkill -9 -f "main[.]py --config" 2>/dev/null; sleep 3
echo "free RAM: $(free -g | awk '/^Mem:/{print $7}') GB | swap used: $(free -g | awk '/^Swap:/{print $3}') GB"

echo "=== 2. building exllamav3 dev (full parallelism) ==="
cd "$EX"
git log --oneline -1
export CUDA_HOME=/usr/local/cuda-13.3 PATH=/usr/local/cuda-13.3/bin:$PATH
export TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=6 CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15
uv pip install --python venv/bin/python -e . --no-build-isolation 2>&1 | tail -3
venv/bin/python -c "import exllamav3; print('BUILD-OK')" || { echo "BUILD-FAILED"; exit 1; }

echo "=== 3. host RAM setting ==="
# 8192 was too aggressive: tensor-parallel duplicates host-side state across the
# worker process, and 8 GB of recurrent checkpoints on top pushed the box into
# zram swap (19.6 GB of the server swapped out, CPU burning on decompression).
sed -i 's/^  sysmem_recurrent_cache: 8192$/  sysmem_recurrent_cache: 4096/' "/home/dushyant30suthar/.config/opencode/providers/exl3/models/Qwen3.6-27B-exl3-5.00bpw.yml"
grep -E "sysmem_recurrent_cache|cache_size|tensor_parallel|draft_mode" "/home/dushyant30suthar/.config/opencode/providers/exl3/models/Qwen3.6-27B-exl3-5.00bpw.yml"

echo "=== 4. restarting ==="
cd "$TB"
nohup "$EX/venv/bin/python" main.py --config /home/dushyant30suthar/.config/opencode/providers/exl3/models/Qwen3.6-27B-exl3-5.00bpw.yml > "$ST/server.log" 2>&1 &
echo $! > "$ST/server.pid"
for i in $(seq 1 60); do
  curl -s --max-time 2 http://127.0.0.1:5000/health 2>/dev/null | grep -q healthy && { echo "HEALTHY after ~$((i*5))s"; break; }
  kill -0 "$(cat $ST/server.pid)" 2>/dev/null || { echo "DIED:"; tail -6 "$ST/server.log"; exit 1; }
  sleep 5
done

echo "=== 5. verify ==="
curl -s --max-time 90 http://127.0.0.1:5000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-27B-exl3-5.00bpw","messages":[{"role":"user","content":"list /tmp with the bash tool /no_think"}],"max_tokens":200,"tools":[{"type":"function","function":{"name":"bash","description":"run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]}' \
  | python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; print('tool_calls OK:', bool(m.get('tool_calls')))"
echo "VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
echo "RAM: $(free -g | awk '/^Mem:/{printf "%s used, %s available", $3, $7}') | swap $(free -g | awk '/^Swap:/{print $3}') GB"
echo "UPGRADE-COMPLETE"
