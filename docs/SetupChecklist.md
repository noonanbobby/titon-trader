# PROJECT TITAN: Operator Setup Checklist
## Complete this while Claude Code builds the system

---

## PHASE 1: IMMEDIATE (Do These First — Day 1)

### 1.1 WSL Ubuntu Environment
You should already have WSL2 with Ubuntu. Verify and update:

```bash
# In Windows PowerShell (Admin):
wsl --update
wsl --set-default-version 2

# Open Ubuntu terminal and update:
sudo apt update && sudo apt upgrade -y
```

### 1.2 Install Docker Desktop for Windows
- Download from: https://www.docker.com/products/docker-desktop/
- During install: **Enable WSL 2 backend** (critical)
- After install: Open Docker Desktop → Settings → Resources → WSL Integration → Enable for your Ubuntu distro
- Verify in Ubuntu terminal:
```bash
docker --version        # Should show 24.x+
docker compose version  # Should show v2.x+
```

### 1.3 Install Development Tools in WSL Ubuntu
```bash
# Python 3.12
sudo apt install -y python3.12 python3.12-venv python3.12-dev python3-pip

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Git
sudo apt install -y git
git config --global user.name "Bobby"
git config --global user.email "your-email@example.com"

# Build essentials (needed for some Python packages)
sudo apt install -y build-essential libssl-dev libffi-dev

# Node.js (needed for some MCP servers)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Useful tools
sudo apt install -y curl wget jq htop tmux
```

### 1.4 Install Claude Code
```bash
# In WSL Ubuntu:
npm install -g @anthropic-ai/claude-code

# Verify:
claude --version

# Login:
claude login
```

### 1.5 Create Project Directory
```bash
mkdir -p ~/projects/titan
cd ~/projects/titan
git init
```

### 1.6 Set Up the .env File
```bash
cd ~/projects/titan
cp .env.example .env
nano .env  # Fill in all values — see below for how to get each one
```

---

## PHASE 2: API KEYS AND SUBSCRIPTIONS (Do These Day 1–2)

### 2.1 Interactive Brokers — CRITICAL
**If you don't already have paper trading enabled:**
1. Log into IBKR Account Management: https://www.interactivebrokers.com/sso/Login
2. Go to Settings → Paper Trading Account → Enable
3. Note your paper trading credentials (same username, separate password)

**Enable API Access:**
1. If you have TWS installed: Configure → API → Settings
   - Enable ActiveX and Socket Clients: ✅
   - Socket port: 4002 (paper) / 4001 (live)
   - Allow connections from localhost only: ✅
   - Read-only API: ❌ (we need to trade!)
2. If using IB Gateway (our approach): The Docker container handles this

**Verify Market Data Subscriptions (IBKR Account Management → Settings → Market Data Subscriptions):**
- US Securities Snapshot and Futures Value Bundle: ~$10/month (if not waived by commissions)
- US Equity and Options Add-On Streaming Bundle: ~$4.50/month
- OPRA (US Options Exchanges): ~$1.50/month
- **Total: ~$6–16/month** (waived if you generate $30+/month in commissions)

**Verify Options Trading Permissions:**
- Account Management → Settings → Trading Permissions
- Ensure "United States" → "Options" is enabled
- Ensure options knowledge level is "Good" or "Extensive"
- You need at minimum Level 2 (spreads) — with $150K you qualify for Level 3+

### 2.2 Anthropic (Claude API Key) — CRITICAL
1. Go to: https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API Keys → Create Key
4. **Copy the key immediately** (starts with `sk-ant-`)
5. Add credits: Settings → Billing → Add funds ($50 minimum to start, recommend $100)
6. **Note:** This is separate from your Claude.ai subscription. The API is pay-per-use
7. Save as `ANTHROPIC_API_KEY` in your `.env`

### 2.3 Polygon.io — HIGH PRIORITY
1. Go to: https://polygon.io/
2. Sign up → Choose **Stocks Advanced** plan ($199/month)
   - This includes: real-time data, options data, historical data, dark pool, news
3. Navigate to Dashboard → API Keys
4. Copy your API key
5. Save as `POLYGON_API_KEY` in your `.env`
6. **Alternative for testing:** Free tier gives 5 API calls/minute — enough to start developing

### 2.4 Unusual Whales — HIGH PRIORITY
1. Go to: https://unusualwhales.com/
2. Sign up for a subscription ($50/month minimum for data access)
3. Then add API access: https://unusualwhales.com/public-api
4. Generate API key
5. Save as `UNUSUAL_WHALES_API_KEY` in your `.env`

### 2.5 Finnhub — FREE
1. Go to: https://finnhub.io/
2. Sign up (free account gives 60 calls/minute — sufficient)
3. Dashboard → API Key
4. Save as `FINNHUB_API_KEY` in your `.env`

### 2.6 Quiver Quantitative — HIGH VALUE
1. Go to: https://www.quiverquant.com/
2. Sign up for API access ($10–25/month)
3. Get API key from dashboard
4. Save as `QUIVER_API_KEY` in your `.env`

### 2.7 FRED (Federal Reserve Economic Data) — FREE
1. Go to: https://fred.stlouisfed.org/
2. Create account → My Account → API Keys → Request API Key
3. Save as `FRED_API_KEY` in your `.env`

### 2.8 SEC EDGAR — FREE (No API Key)
- No registration needed for EDGAR XBRL data
- We parse Form 4 XML directly from: https://www.sec.gov/cgi-bin/browse-edgar
- Set a User-Agent header: "Titan Trading Bot (your-email@example.com)"

---

## PHASE 3: NOTIFICATION SETUP (Day 2)

### 3.1 Telegram Bot — CRITICAL FOR MONITORING
1. Open Telegram → Search for `@BotFather`
2. Send `/newbot`
3. Name it: `Titan Trading Bot` (or whatever you prefer)
4. Username: `titan_trading_xxxxxx_bot` (must be unique)
5. **Copy the bot token** → Save as `TELEGRAM_BOT_TOKEN` in `.env`
6. To get your Chat ID:
   - Send any message to your new bot
   - Visit: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
   - Find `"chat":{"id":XXXXXXXX}` — that number is your chat ID
7. Save as `TELEGRAM_CHAT_ID` in `.env`
8. **Test it:** In a browser, visit:
   `https://api.telegram.org/bot<TOKEN>/sendMessage?chat_id=<CHAT_ID>&text=Titan+Online`

### 3.2 Twilio SMS (For Critical Alerts)
1. Go to: https://www.twilio.com/
2. Sign up (free trial gives $15 credit — enough for months of alerts)
3. Get a phone number (free with trial)
4. Dashboard → Account Info:
   - Account SID → `TWILIO_ACCOUNT_SID`
   - Auth Token → `TWILIO_AUTH_TOKEN`
   - Your Twilio number → `TWILIO_FROM_NUMBER`
5. Your personal cell → `TWILIO_TO_NUMBER`
6. **Note:** Free trial requires verifying your personal number first

---

## PHASE 4: VERIFY IBKR CONNECTIVITY (Day 1, After Docker Is Running)

Once Claude Code has the Docker Compose stack running, verify IB Gateway:

### 4.1 Start the Stack
```bash
cd ~/projects/titan
docker compose up -d
```

### 4.2 Check IB Gateway
```bash
# Check if gateway is running
docker compose logs ib-gateway

# If you need to see the GUI (for 2FA or debugging):
# Install a VNC viewer and connect to localhost:5900
```

### 4.3 First-Time 2FA
- The first time IB Gateway starts, you may need to complete 2FA
- Option A: Use IBKR Mobile app (push notification)
- Option B: Connect via VNC to see the Gateway GUI
- IBC will handle subsequent logins automatically

### 4.4 Verify API Connection
```python
# Quick test script (run in the titan container or locally)
import asyncio
from ib_async import IB

async def test():
    ib = IB()
    await ib.connectAsync('127.0.0.1', 4002, clientId=99)
    print(f"Connected: {ib.isConnected()}")
    account = ib.managedAccounts()
    print(f"Account: {account}")
    ib.disconnect()

asyncio.run(test())
```

---

## PHASE 5: VERIFY DATA SOURCES (Day 2–3)

### 5.1 Test Each API
Run these quick tests to verify all API keys work:

```python
# Polygon.io
import httpx
r = httpx.get("https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey=YOUR_KEY")
print(r.json())

# Finnhub
r = httpx.get("https://finnhub.io/api/v1/quote?symbol=AAPL&token=YOUR_KEY")
print(r.json())

# FRED
r = httpx.get("https://api.stlouisfed.org/fred/series/observations?series_id=T10Y2Y&api_key=YOUR_KEY&file_type=json&sort_order=desc&limit=5")
print(r.json())

# Unusual Whales
r = httpx.get("https://api.unusualwhales.com/api/stock/AAPL/options-volume",
              headers={"Authorization": "Bearer YOUR_KEY"})
print(r.json())

# Claude API
from anthropic import Anthropic
client = Anthropic(api_key="YOUR_KEY")
message = client.messages.create(model="claude-sonnet-4-5-20250929", max_tokens=100,
    messages=[{"role": "user", "content": "Say 'Titan Online' if you can read this."}])
print(message.content[0].text)
```

---

## PHASE 6: GRAFANA DASHBOARD ACCESS (Day 5)

Once the stack is running with Grafana:

1. Open browser: http://localhost:3000
2. Login: admin / (your GRAFANA_PASSWORD from .env, default: admin)
3. Dashboards should auto-provision from the JSON config
4. Verify data sources are connected:
   - PostgreSQL → titan database
   - QuestDB → time-series data
   - Prometheus → system metrics

---

## PHASE 7: PRE-LIVE TRADING CHECKLIST (Day 6–7)

Before switching from paper to live trading:

### ✅ System Verification
- [ ] IB Gateway connects reliably to paper account
- [ ] Market data streams without gaps
- [ ] Options chains load with Greeks for all 10 universe tickers
- [ ] At least one test trade (bull call spread) placed and filled on paper
- [ ] At least one test trade closed (manually or by exit criteria)
- [ ] Circuit breakers tested (simulate a losing trade)
- [ ] Telegram notifications received for trade entry/exit
- [ ] Grafana dashboard shows P&L and positions correctly
- [ ] All 10 strategies coded and entry/exit criteria validated
- [ ] Regime detection producing sensible classifications
- [ ] ML ensemble producing confidence scores

### ✅ Risk Verification
- [ ] Per-trade risk capped at $3,000 (verified in position sizer)
- [ ] Max positions capped at 8 (verified in risk manager)
- [ ] Sector concentration limits active
- [ ] Circuit breaker state persists across restarts
- [ ] Emergency stop via Telegram /kill command works

### ✅ Go-Live Steps
1. **Change .env:** `IBKR_TRADING_MODE=live` and `IBKR_GATEWAY_PORT=4001`
2. **Restart stack:** `docker compose down && docker compose up -d`
3. **Complete 2FA** for live account
4. **Start with 1–2 contracts per position** (override position sizer temporarily)
5. **Monitor actively for first 3 trading days**
6. **Scale to normal sizing** after 5+ successful trades

---

## PHASE 8: FUTURE — VPS DEPLOYMENT (Week 3+)

When you're ready to move to always-on VPS:

### 8.1 VPS Provider Selection
**Recommended:** QuantVPS New York or Hetzner Ashburn, VA
- QuantVPS: $59.99/mo for 4 vCPU, 8GB RAM — request Linux/Ubuntu
- Hetzner: CPX31 ~$15/mo for 4 vCPU, 8GB RAM in Ashburn
- **Important:** If your QuantVPS is in Amsterdam, consider getting a US-based VPS instead. Amsterdam adds 75–120ms latency to US exchanges

### 8.2 VPS Setup
```bash
# On the VPS:
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose-v2 git ufw fail2ban

# Security hardening:
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw enable

# Clone your repo:
git clone https://github.com/YOUR_USERNAME/titan.git
cd titan
cp .env.example .env
nano .env  # Fill in production values (IBKR_TRADING_MODE=live)

# Start:
docker compose up -d
```

### 8.3 Remote Access to Grafana
```bash
# Option A: SSH tunnel (simple)
ssh -L 3000:localhost:3000 user@your-vps-ip
# Then visit http://localhost:3000 in your browser

# Option B: Cloudflare Tunnel (zero trust, recommended for production)
# Install cloudflared on VPS, create tunnel, route to localhost:3000
```

---

## COST SUMMARY

### Monthly Recurring Costs
| Item | Cost |
|------|------|
| IBKR Market Data | ~$6–16/mo (often waived) |
| Polygon.io Advanced | $199/mo |
| Unusual Whales + API | ~$50–75/mo |
| Quiver Quantitative | $10–25/mo |
| Claude API (estimated) | $50–100/mo |
| VPS (when deployed) | $60–100/mo |
| Twilio SMS | ~$5/mo |
| **Total** | **~$380–520/mo** |

### One-Time / Free
| Item | Cost |
|------|------|
| Finnhub | Free |
| FRED | Free |
| SEC EDGAR | Free |
| StockTwits | Free |
| Docker | Free |
| Grafana | Free |
| All ML libraries | Free |

### ROI Justification
At even a conservative 5% monthly return on $150,000 = $7,500/month in profit.
Monthly costs of ~$450 represent **6% of projected monthly profit**.
The data and tools pay for themselves many times over.

---

## QUICK REFERENCE: YOUR .env FILE

```bash
# ============================================
# PROJECT TITAN ENVIRONMENT CONFIGURATION
# ============================================

# --- IBKR ---
IBKR_USERNAME=           # Your IBKR username
IBKR_PASSWORD=           # Your IBKR password
IBKR_TRADING_MODE=paper  # START WITH PAPER! Change to 'live' when ready
IBKR_GATEWAY_PORT=4002   # 4002=paper, 4001=live
IBKR_CLIENT_ID=1

# --- Databases ---
POSTGRES_DB=titan
POSTGRES_USER=titan
POSTGRES_PASSWORD=        # Generate: openssl rand -hex 24
REDIS_HOST=redis
REDIS_PORT=6379
QUESTDB_HOST=questdb

# --- API Keys ---
ANTHROPIC_API_KEY=        # From console.anthropic.com
POLYGON_API_KEY=          # From polygon.io dashboard
UNUSUAL_WHALES_API_KEY=   # From unusualwhales.com
FINNHUB_API_KEY=          # From finnhub.io (free)
QUIVER_API_KEY=           # From quiverquant.com
FRED_API_KEY=             # From fred.stlouisfed.org (free)

# --- Notifications ---
TELEGRAM_BOT_TOKEN=       # From @BotFather
TELEGRAM_CHAT_ID=         # Your personal chat ID
TWILIO_ACCOUNT_SID=       # From twilio.com
TWILIO_AUTH_TOKEN=        # From twilio.com
TWILIO_FROM_NUMBER=       # Your Twilio number
TWILIO_TO_NUMBER=         # Your personal cell

# --- Trading Parameters ---
ACCOUNT_SIZE=150000
MAX_DRAWDOWN_PCT=0.15
PER_TRADE_RISK_PCT=0.02
MAX_CONCURRENT_POSITIONS=8
CONFIDENCE_THRESHOLD=0.78

# --- Claude AI ---
CLAUDE_MODEL=claude-sonnet-4-5-20250929
CLAUDE_ANALYSIS_THINKING_BUDGET=8192
CLAUDE_RISK_THINKING_BUDGET=4096

# --- Grafana ---
GRAFANA_PASSWORD=          # Generate: openssl rand -hex 12
```

---

*Complete this checklist in parallel with Claude Code's build. By Day 7, you should have all accounts configured, all API keys tested, and the system ready for live trading.*
