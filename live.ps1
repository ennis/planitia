param([string]$profile = "release")

cargo build -p game -p hot-reload-test --profile $profile
Start-Process "wt" "-d . cargo run -p game --profile $profile"
bacon -j hot-reload
