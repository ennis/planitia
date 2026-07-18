cargo build -p game --release
Start-Process -NoNewWindow "cargo" "run -p game --release"
bacon -j hot-reload
