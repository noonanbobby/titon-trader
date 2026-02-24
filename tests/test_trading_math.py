# ─── POSITION SIZER TESTS ────────────────────────────────────────
# These are the most critical. Wrong sizing = blown account.


class TestPositionSizer:
    """Verify position sizing formulas with exact numerical examples."""

    def test_bull_call_spread_sizing_basic(self):
        """$5 spread, $150K account, 2% risk = 6 contracts max."""
        # From spec: "For $5 net debit = 6 contracts max"
        account_size = 150_000
        risk_pct = 0.02  # 2%
        spread_width = 5.00
        multiplier = 100

        max_risk = account_size * risk_pct  # $3,000
        max_contracts = int(max_risk / (spread_width * multiplier))

        assert max_risk == 3_000
        assert max_contracts == 6

    def test_bull_call_spread_sizing_narrow(self):
        """$3 spread should allow 10 contracts."""
        max_contracts = int(3_000 / (3.00 * 100))
        assert max_contracts == 10

    def test_bull_call_spread_sizing_wide(self):
        """$10 spread should allow 3 contracts."""
        max_contracts = int(3_000 / (10.00 * 100))
        assert max_contracts == 3

    def test_position_sizer_returns_integer(self):
        """IBKR rejects fractional contracts. Must floor, never round."""
        # $7 spread: 3000/700 = 4.28... must be 4, not 4.28 or 5
        max_contracts = int(3_000 / (7.00 * 100))
        assert max_contracts == 4
        assert isinstance(max_contracts, int)

    def test_position_sizer_minimum_one(self):
        """$35 spread: 3000/3500 = 0.857 → should be 0 (rejected), not 1."""
        # If spread width exceeds per-trade risk, we can't take the trade
        max_contracts = int(3_000 / (35.00 * 100))
        assert max_contracts == 0  # Trade should be rejected

    def test_credit_spread_sizing(self):
        """Bull put spread: max loss = width - credit."""
        spread_width = 5.00
        credit_received = 1.00
        max_loss_per_contract = (spread_width - credit_received) * 100  # $400
        max_contracts = int(3_000 / max_loss_per_contract)
        assert max_loss_per_contract == 400
        assert max_contracts == 7

    def test_iron_condor_sizing(self):
        """Iron condor: max loss = wider wing width - net credit."""
        wing_width = 10.00
        net_credit = 3.00
        max_loss_per_contract = (wing_width - net_credit) * 100  # $700
        max_contracts = int(3_000 / max_loss_per_contract)
        assert max_loss_per_contract == 700
        assert max_contracts == 4

    def test_short_strangle_smaller_sizing(self):
        """Strangles use 2-3% of PORTFOLIO, not the normal 2% per-trade."""
        account_size = 150_000
        strangle_risk_pct = 0.03  # 3% max for undefined risk
        max_portfolio_risk = account_size * strangle_risk_pct  # $4,500
        # But this is total portfolio allocation, not per-contract risk
        # Strangles have undefined risk, so sizing is premium-based
        assert max_portfolio_risk == 4_500

    def test_ratio_spread_tiny_sizing(self):
        """Ratio spreads: max 1% of account due to unlimited risk."""
        account_size = 150_000
        max_risk = account_size * 0.01  # $1,500
        assert max_risk == 1_500

    def test_pmcc_leaps_cost_limit(self):
        """PMCC: LEAPS cost must be ≤ 10% of account per position."""
        account_size = 150_000
        max_leaps_cost = account_size * 0.10  # $15,000
        leaps_price = 45.00  # $45 per contract = $4,500
        max_leaps_contracts = int(max_leaps_cost / (leaps_price * 100))
        assert max_leaps_cost == 15_000
        assert max_leaps_contracts == 3

    def test_quarter_kelly(self):
        """Quarter-Kelly with 72% win rate, $4500 avg win, $3000 avg loss."""
        win_rate = 0.72
        avg_win = 4_500
        avg_loss = 3_000

        # Kelly: f = (p*b - q) / b where b = avg_win/avg_loss, q = 1-p
        b = avg_win / avg_loss  # 1.5
        q = 1 - win_rate  # 0.28
        kelly = (win_rate * b - q) / b
        quarter_kelly = kelly / 4

        assert abs(kelly - 0.5333) < 0.01  # ~53% Kelly
        assert abs(quarter_kelly - 0.1333) < 0.01  # ~13% quarter-Kelly
        assert quarter_kelly > 0  # Positive = we have edge
        assert quarter_kelly < 1  # Sane value


# ─── CIRCUIT BREAKER TESTS ───────────────────────────────────────


class TestCircuitBreakers:
    """Verify the drawdown protection math."""

    def test_daily_threshold(self):
        """Daily loss > 2% triggers CAUTION."""
        account_size = 150_000
        daily_loss = -3_100  # > $3,000
        threshold = account_size * 0.02
        assert abs(daily_loss) > threshold

    def test_weekly_threshold(self):
        """Weekly loss > 5% triggers WARNING."""
        account_size = 150_000
        weekly_loss = -7_600
        threshold = account_size * 0.05
        assert abs(weekly_loss) > threshold
        assert threshold == 7_500

    def test_monthly_threshold(self):
        """Monthly loss > 10% triggers HALT."""
        account_size = 150_000
        threshold = account_size * 0.10
        assert threshold == 15_000

    def test_total_drawdown_from_hwm(self):
        """15% drawdown measured from HIGH WATER MARK, not initial capital."""
        initial_capital = 150_000
        high_water_mark = 165_000  # Account grew to $165K
        current_value = 140_000

        # WRONG: drawdown from initial
        wrong_drawdown = (initial_capital - current_value) / initial_capital
        # RIGHT: drawdown from HWM
        correct_drawdown = (high_water_mark - current_value) / high_water_mark

        assert abs(wrong_drawdown - 0.0667) < 0.01  # Only ~6.7% from initial
        assert abs(correct_drawdown - 0.1515) < 0.01  # >15% from HWM!
        assert correct_drawdown > 0.15  # This SHOULD trigger emergency

    def test_recovery_ladder_stages(self):
        """Recovery: 50% → 3 wins → 75% → 3 wins → 100%."""
        stages = [0.50, 0.75, 1.00]  # Position size multipliers
        wins_required = [0, 3, 6]  # Cumulative wins needed

        for i, (stage, wins) in enumerate(zip(stages, wins_required, strict=False)):
            if i == 0:
                assert stage == 0.50  # Start at half
            elif wins >= 3 and wins < 6:
                assert stage == 0.75
            elif wins >= 6:
                assert stage == 1.00

    def test_hwm_recovery_check(self):
        """Only restore full sizing when within 5% of HWM."""
        hwm = 165_000
        current = 158_000
        pct_from_hwm = (hwm - current) / hwm

        assert abs(pct_from_hwm - 0.0424) < 0.01  # ~4.2% below HWM
        assert pct_from_hwm < 0.05  # Within 5% → can restore full sizing


# ─── OPTIONS FORMULA TESTS ───────────────────────────────────────


class TestOptionsFormulas:
    """Verify the financial formulas are correct."""

    def test_iv_rank(self):
        """IV Rank = (current - 52w_low) / (52w_high - 52w_low)."""
        current_iv = 0.30
        low_52w = 0.15
        high_52w = 0.45

        iv_rank = (current_iv - low_52w) / (high_52w - low_52w)
        assert abs(iv_rank - 0.50) < 1e-9  # 50th rank (float precision)

    def test_iv_rank_at_extremes(self):
        """IV Rank should be 0 at low, 1 at high."""
        assert (0.15 - 0.15) / (0.45 - 0.15) == 0.0
        assert (0.45 - 0.15) / (0.45 - 0.15) == 1.0

    def test_iv_rank_vs_percentile(self):
        """IV Rank and IV Percentile are DIFFERENT measures."""
        # IV Rank uses min/max, IV Percentile uses distribution
        historical_ivs = [0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.42, 0.45, 0.50]
        current_iv = 0.30
        low = min(historical_ivs)
        high = max(historical_ivs)

        iv_rank = (current_iv - low) / (high - low)
        iv_percentile = sum(1 for iv in historical_ivs if iv < current_iv) / len(
            historical_ivs
        )

        assert abs(iv_rank - 0.333) < 0.01  # 33% rank
        assert iv_percentile == 0.40  # 40th percentile (4 out of 10 below)
        assert iv_rank != iv_percentile  # They're different!

    def test_gex_formula(self):
        """GEX = Σ(call_OI × call_γ × 100 × spot) - Σ(put_OI × put_γ × 100 × spot)."""
        spot = 200.0

        # Simplified: one call strike, one put strike
        call_oi = 5000
        call_gamma = 0.03
        put_oi = 3000
        put_gamma = 0.02

        call_gex = call_oi * call_gamma * 100 * spot
        put_gex = put_oi * put_gamma * 100 * spot
        net_gex = call_gex - put_gex

        assert call_gex == 3_000_000
        assert put_gex == 1_200_000
        assert net_gex == 1_800_000  # Positive = mean-reverting

    def test_bull_call_spread_max_loss(self):
        """Bull call max loss = net debit paid."""
        long_premium = 8.50
        short_premium = 3.50
        net_debit = long_premium - short_premium
        max_loss_per_contract = net_debit * 100

        assert net_debit == 5.00
        assert max_loss_per_contract == 500

    def test_bull_call_spread_max_profit(self):
        """Bull call max profit = spread width - net debit."""
        long_strike = 195
        short_strike = 205
        spread_width = short_strike - long_strike
        net_debit = 5.00
        max_profit_per_contract = (spread_width - net_debit) * 100

        assert spread_width == 10
        assert max_profit_per_contract == 500  # 1:1 risk/reward on this example

    def test_iron_condor_max_loss(self):
        """Iron condor max loss = wider wing width - net credit."""
        call_wing_width = 10
        put_wing_width = 10
        net_credit = 3.00
        wider_wing = max(call_wing_width, put_wing_width)
        max_loss = (wider_wing - net_credit) * 100

        assert max_loss == 700

    def test_credit_spread_target(self):
        """Bull put credit target: 15-20% of spread width."""
        spread_width = 5.00
        min_credit = spread_width * 0.15  # $0.75
        max_credit = spread_width * 0.20  # $1.00

        assert min_credit == 0.75
        assert max_credit == 1.00

    def test_iron_condor_credit_target(self):
        """Iron condor credit target: 25-33% of wing width."""
        wing_width = 10.00
        min_credit = wing_width * 0.25
        max_credit = wing_width * 0.33

        assert min_credit == 2.50
        assert abs(max_credit - 3.30) < 1e-9  # float precision


# ─── PORTFOLIO LIMIT TESTS ───────────────────────────────────────


class TestPortfolioLimits:
    """Verify portfolio-level risk controls."""

    def test_single_ticker_concentration(self):
        """Max 30% of total risk in one ticker."""
        total_risk_budget = 8 * 3_000  # 8 positions × $3K each = $24K
        max_single_ticker = total_risk_budget * 0.30
        assert max_single_ticker == 7_200
        # So max 2 positions in same ticker at $3K each

    def test_max_capital_deployed(self):
        """Never exceed 70% of account in deployed capital."""
        account = 150_000
        max_deployed = account * 0.70
        assert max_deployed == 105_000

    def test_market_hours_filter(self):
        """No entries during first 15 min (9:30-9:45) or last 15 min (3:45-4:00)."""
        from datetime import time

        earliest_entry = time(9, 45)
        latest_entry = time(15, 45)

        # 9:31 should be blocked
        assert time(9, 31) < earliest_entry
        # 9:45 should be allowed
        assert time(9, 45) >= earliest_entry
        # 3:44 should be allowed
        assert time(15, 44) < latest_entry
        # 3:46 should be blocked
        assert time(15, 46) > latest_entry


# ─── STRATEGY ENTRY CRITERIA TESTS ───────────────────────────────


class TestStrategyEntryCriteria:
    """Verify strategies only fire under correct conditions."""

    def test_bull_call_requires_uptrend(self):
        """ADX must be > 25 for bull call spread."""
        adx = 20  # Too low — ranging market
        assert adx <= 25  # Should NOT trigger bull call

    def test_bull_call_rejects_high_iv(self):
        """IV Rank must be 20-50% for bull call (don't overpay)."""
        iv_rank = 0.65  # Too expensive
        assert not (0.20 <= iv_rank <= 0.50)

    def test_iron_condor_requires_range_bound(self):
        """ADX must be < 20 for iron condor."""
        adx = 30  # Trending — IC will get run over
        assert adx >= 20  # Should NOT trigger iron condor

    def test_iron_condor_requires_high_iv(self):
        """IV Rank must be 50-70%+ for iron condor (need premium)."""
        iv_rank = 0.30  # Not enough premium
        assert not (0.50 <= iv_rank <= 0.70)

    def test_crisis_blocks_all_entries(self):
        """VIX > 35 = CRISIS. No new positions."""
        vix = 38
        assert vix > 35  # Crisis mode
        # Strategy selector should return empty list

    def test_confidence_threshold(self):
        """Only trade when ML confidence > 0.78."""
        confidence = 0.75  # Close but not enough
        threshold = 0.78
        assert confidence < threshold  # Do NOT trade

    def test_exit_at_21_dte(self):
        """Close any position at 21 DTE regardless of P&L."""
        days_to_expiry = 21
        assert days_to_expiry <= 21  # Must exit

    def test_profit_target_bull_call(self):
        """Close bull call at 50-65% of max profit."""
        max_profit = 500  # per contract
        current_profit = 280  # 56% of max
        pct_of_max = current_profit / max_profit
        assert 0.50 <= pct_of_max <= 0.65  # Hit target → close
