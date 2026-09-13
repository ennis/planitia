param([string]$profile = "release")

cargo build -p game-host -p experiment --profile $profile
Start-Process "wt" "-d . cargo run -p game-host --profile $profile"
bacon --headless -j hot-reload -- $profile
