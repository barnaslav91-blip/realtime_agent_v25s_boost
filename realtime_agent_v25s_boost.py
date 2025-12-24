import yaml
with open("config_boost.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
mode = cfg.get("mode", "BOOST")
risk_per_trade = cfg.get("risk_per_trade", 0.01)
max_daily_drawdown = cfg.get("max_daily_drawdown", 0.05)
max_equity_drawdown = cfg.get("max_equity_drawdown", 0.30)
signals_file = cfg.get("signals_file", "signals_boost.csv")
log_file = cfg.get("log_file", "trades_boost.log")
import time
from datetime import datetime, date
from typing import List, Tuple

import MetaTrader5 as mt5
import pandas as pd
import joblib

from agent_scalper_v25s import make_signal, calc_rsi

print("Реальный агент v1.0 (V25(1s) M5, AI + MedianRange SL/TP + Equity Guard) запущен.")

# =======================
# НАСТРОЙКИ
# =======================

# Символ в MT5 (проверь точное имя в терминале)
SYMBOL = "Volatility 25 (1s) Index"

# Таймфрейм
TIMEFRAME = mt5.TIMEFRAME_M5

# Режим риска: "conservative" | "standard" | "aggressive" | "BOOST"
RISK_MODE = "BOOST"  # для этого файла логично сразу поставить BOOST

RISK_PER_TRADE_BY_MODE = {
    "conservative": 0.0025,
    "standard":     0.0050,
    "aggressive":   0.0100,
}

if RISK_MODE == "BOOST":
    RISK_PER_TRADE = risk_per_trade  # берём из config_boost.yaml
else:
    if RISK_MODE not in RISK_PER_TRADE_BY_MODE:
        raise ValueError(f"Неизвестный RISK_MODE={RISK_MODE}. Допустимо: {list(RISK_PER_TRADE_BY_MODE.keys())}")
    RISK_PER_TRADE = RISK_PER_TRADE_BY_MODE[RISK_MODE]

print(f"[CONFIG] mode={mode}, RISK_MODE={RISK_MODE}, risk_per_trade={RISK_PER_TRADE:.4f}")

# AI-модель
MODEL_PATH = r"C:\Users\Vaslav91\AITrader\ai_filter_v25s_model.pkl"
SCALER_PATH = r"C:\Users\Vaslav91\AITrader\ai_filter_v25s_scaler.pkl"
AI_THRESHOLD = 0.46

# SL/TP через MedianRange
SL_MULT = 3.0
TP_MULT = 6.0

# История для сигналов
LOOKBACK_BARS = 400
MIN_BARS_BETWEEN_TRADES = 5

# Equity Guard
MAX_DRAWDOWN_PCT = max_equity_drawdown      # из config_boost.yaml (0.30)
MAX_DAILY_LOSS_PCT = max_daily_drawdown     # из config_boost.yaml (0.08)

# Magic для позиций
MAGIC = 26001

# Путь для логов (по желанию)
TRADES_LOG_PATH = log_file

# === Фичи для AI (как в train_ai_filter_v25s.py) ===
FEATURE_COLS = [
    "direction",
    "price",
    "ema_fast",
    "trend_ema",
    "rsi",
    "range",
    "median_range",
    "trend_flag",
]


# =======================
# ИНИЦИАЛИЗАЦИЯ MT5, МОДЕЛИ
# =======================

def init_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"Не удалось инициализировать MetaTrader5: {mt5.last_error()}")
    print("[MT5] Инициализация успешна.")

    info = mt5.account_info()
    if info is None:
        raise RuntimeError("Не удалось получить account_info. Проверь, что терминал залогинен в ДЕМО-счёт.")
    print(f"[MT5] Аккаунт: #{info.login}, balance={info.balance}, equity={info.equity}, server={info.server}")

    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        raise RuntimeError(f"Символ {SYMBOL} не найден. Добавь его в Market Watch.")
    if not symbol_info.visible:
        if not mt5.symbol_select(SYMBOL, True):
            raise RuntimeError(f"Не удалось сделать символ {SYMBOL} видимым.")
    print(f"[MT5] SYMBOL={SYMBOL}, digits={symbol_info.digits}, point={symbol_info.point}")

    return info, symbol_info


def load_ai():
    print("[AI] Загружаем модель:", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    print("[AI] Загружаем скейлер:", SCALER_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


# =======================
# УТИЛИТЫ
# =======================

def get_rates_df(count: int) -> pd.DataFrame:
    """Забираем последние count баров и приводим к формату DataFrame."""
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"[MT5] Не удалось получить котировки для {SYMBOL}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    # 'time' -> datetime
    df["DateTime"] = pd.to_datetime(df["time"], unit="s")
    df["Date"] = df["DateTime"].dt.date
    df["Time"] = df["DateTime"].dt.time
    df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "tick_volume": "TICKVOL",
        "real_volume": "VOL",
        "spread": "SPREAD"
    }, inplace=True)
    return df

def get_current_position():
    """Возвращает (position_dir, volume, entry_price, sl, tp, ticket) по SYMBOL и нашему MAGIC."""
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None or len(positions) == 0:
        return 0, 0.0, None, None, None, None

    for pos in positions:
        # Берём только свои позиции (по MAGIC) 
        if pos.magic == MAGIC:
            direction = 1 if pos.type == mt5.POSITION_TYPE_BUY else -1
            return direction, pos.volume, pos.price_open, pos.sl, pos.tp, pos.ticket

    # Если по нашему MAGIC ничего нет — считаем, что позиции нет
    return 0, 0.0, None, None, None, None

def send_order(direction: int, volume: float, sl_price: float, tp_price: float, comment: str = ""):
    """Открытие рыночного ордера с SL/TP."""
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        raise RuntimeError(f"[ORDER] Не найден SYMBOL={SYMBOL}")

    # Ограничим volume под допуски символа
    vol_min = symbol_info.volume_min
    vol_max = symbol_info.volume_max
    vol_step = symbol_info.volume_step

    # Приводим к шагу
    volume = max(vol_min, min(vol_max, volume))
    volume = round(volume / vol_step) * vol_step

    if volume <= 0:
        print("[ORDER] Volume <= 0, ордер не отправлен")
        return None

    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(SYMBOL).ask if direction == 1 else mt5.symbol_info_tick(SYMBOL).bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": 30,
        "magic": MAGIC,
        "comment": comment,
        "type_filling": mt5.ORDER_FILLING_FOK,
        "type_time": mt5.ORDER_TIME_GTC,
    }

    print(f"[ORDER] Отправка ордера: dir={direction}, vol={volume}, price={price}, SL={sl_price}, TP={tp_price}")
    result = mt5.order_send(request)
    if result is None:
        print("[ORDER] order_send вернул None:", mt5.last_error())
        return None

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ORDER] Ошибка выполнения, retcode={result.retcode}, details={result}")
        return None

    print(f"[ORDER] Открыта позиция ticket={result.order}, volume={volume}")
    return result


def log_trade(row: dict):
    """Простейшее логирование сделок в CSV."""
    df_row = pd.DataFrame([row])
    try:
        # append mode
        df_row.to_csv(TRADES_LOG_PATH, mode="a", header=not pd.io.common.file_exists(TRADES_LOG_PATH), index=False)
    except Exception as e:
        print(f"[LOG] Ошибка записи лога: {e}")


# =======================
# ОСНОВНОЙ ЦИКЛ АГЕНТА
# =======================

def run_agent():
    account_info, symbol_info = init_mt5()
    model, scaler = load_ai()

    balance = account_info.balance
    equity = account_info.equity
    equity_peak = equity
    current_date = date.today()
    daily_start_balance = balance

    last_bar_time = None
    last_signal = None
    last_trade_bar_index = None

    print(f"[AGENT] Старт. Balance={balance}, Equity={equity}, дата={current_date}")

    while True:
        try:
            # забираем историю
            df = get_rates_df(LOOKBACK_BARS + 100)
        except Exception as e:
            print(f"[ERROR] При получении котировок: {e}")
            time.sleep(5)
            continue

        if len(df) < LOOKBACK_BARS + 10:
            print("[WARN] Мало истории, ждём...")
            time.sleep(10)
            continue

        # последний бар
        last_row = df.iloc[-1]
        bar_time = last_row["DateTime"]

        # ждём новый бар (работаем по закрытию, поэтому проверяем изменение времени)
        if last_bar_time is not None and bar_time <= last_bar_time:
            time.sleep(5)
            continue

        # обновили время
        last_bar_time = bar_time

        # пересчёт индикаторов на df
        df["Range"] = df["High"] - df["Low"]
        df["MedianRange"] = df["Range"].rolling(window=50).median()
        df["EMA_FAST"] = df["Close"].ewm(span=30, adjust=False).mean()
        df["TREND_EMA"] = df["Close"].ewm(span=100, adjust=False).mean()
        df["RSI"] = calc_rsi(df["Close"], 14)

        # окно сигналов
        window = df.iloc[-LOOKBACK_BARS:]
        median_range = float(window.iloc[-1]["MedianRange"]) if not pd.isna(window.iloc[-1]["MedianRange"]) else 0.0

        # обновим account_info
        account_info = mt5.account_info()
        if account_info is None:
            print("[ERROR] Не удалось обновить account_info.")
            time.sleep(5)
            continue

        balance = account_info.balance
        equity = account_info.equity

        # контроль дня
        bar_date = bar_time.date()
        if bar_date != current_date:
            # зафиксируем дневной результат
            day_loss = daily_start_balance - balance
            if daily_start_balance > 0:
                day_loss_pct = day_loss / daily_start_balance * 100
                print(f"[DAY] День закончился. Day PnL={-day_loss:.2f}, ({-day_loss_pct:.2f}%) от старта дня.")
            current_date = bar_date
            daily_start_balance = balance
            print(f"[DAY] Новый день: {current_date}, daily_start_balance={daily_start_balance:.2f}")

        # обновление equity_peak и просадки
        if equity > equity_peak:
            equity_peak = equity
        dd_pct = 0.0
        if equity_peak > 0:
            dd_pct = (equity_peak - equity) / equity_peak

        # дневной убыток
        day_loss_now = daily_start_balance - balance
        day_loss_now_pct = (day_loss_now / daily_start_balance) if daily_start_balance > 0 else 0.0

        # флаг, можно ли открывать новые сделки
        trading_enabled = True
        if dd_pct >= MAX_DRAWDOWN_PCT:
            trading_enabled = False
        if day_loss_now_pct >= MAX_DAILY_LOSS_PCT:
            trading_enabled = False

        # позиция
        pos_dir, pos_vol, pos_entry, pos_sl, pos_tp, pos_ticket = get_current_position()

        print(
            f"[BAR] {bar_time} | bal={balance:.2f}, eq={equity:.2f}, dd={dd_pct*100:.2f}%, "
            f"dayLoss={day_loss_now_pct*100:.2f}%, pos_dir={pos_dir}, pos_vol={pos_vol}"
        )

        # если защита сработала — новых ВХОДОВ не делаем, но существующую позицию не трогаем
        if not trading_enabled:
            print("[GUARD] Лимит риска достигнут. Новые сделки не открываем.")
            time.sleep(5)
            continue

        # если уже есть позиция — работаем только через SL/TP, новые входы не открываем
        if pos_dir != 0:
            print("[INFO] Позиция уже открыта, ждём SL/TP.")
            time.sleep(5)
            continue

        # если медианный диапазон ещё не посчитан
        if median_range <= 0:
            print("[INFO] MedianRange ещё не посчитан (мало истории), пропускаем бар.")
            time.sleep(5)
            continue

        # генерируем сигнал
        signal = make_signal(window, last_signal)
        last_signal = signal

        if signal not in ("BUY", "SELL"):
            print(f"[SIG] Сигнал={signal}, новых сделок нет.")
            time.sleep(5)
            continue

        # рассчитываем фичи для AI
        row_sig = window.iloc[-1]
        price = float(row_sig["Close"])
        ema_fast = float(row_sig["EMA_FAST"])
        trend_ema = float(row_sig["TREND_EMA"])
        rsi = float(row_sig["RSI"])
        rng = float(row_sig["Range"])
        med_rng = float(row_sig["MedianRange"])

        if price > trend_ema:
            trend_flag = 1
        elif price < trend_ema:
            trend_flag = -1
        else:
            trend_flag = 0

        direction_flag = 1 if signal == "BUY" else -1

        feature_dict = {
            "direction": direction_flag,
            "price": price,
            "ema_fast": ema_fast,
            "trend_ema": trend_ema,
            "rsi": rsi,
            "range": rng,
            "median_range": med_rng,
            "trend_flag": trend_flag,
        }
        features_df = pd.DataFrame([feature_dict], columns=FEATURE_COLS)

        # загрузим модель/скейлер один раз снаружи
        # но здесь используем их
        try:
            model, scaler  # чтобы линтер не ругался
        except NameError:
            # если вдруг не загружено (но мы грузим в run_agent)
            model, scaler = load_ai()

        features_scaled = scaler.transform(features_df)
        prob_good = model.predict_proba(features_scaled)[0, 1]

        print(f"[AI] Сигнал={signal}, prob_good={prob_good:.3f}")

        if prob_good < AI_THRESHOLD:
            print("[AI] Фильтр отклонил сделку.")
            time.sleep(5)
            continue
 # ===== BOOST-ФИЛЬТРЫ =====

        # 1) Не входим против тренда по TREND_EMA
        if signal == "BUY" and price < trend_ema:
            print("[BOOST] BUY против тренда (price < TREND_EMA), пропускаем.")
            time.sleep(5)
            continue

        if signal == "SELL" and price > trend_ema:
            print("[BOOST] SELL против тренда (price > TREND_EMA), пропускаем.")
            time.sleep(5)
            continue

        # 2) Не входим, если рынок слишком узкий (маленький MedianRange)
        min_med_range = symbol_info.point * 2  # минимум 2 тика медианного диапазона
        if med_rng < min_med_range:
            print(f"[BOOST] Слишком узкий рынок (MedianRange={med_rng:.5f}), пропускаем.")
            time.sleep(5)
            continue

        # считаем SL/TP через MedianRange
        sl_distance = med_rng * SL_MULT
        tp_distance = med_rng * TP_MULT

        if sl_distance <= 0 or tp_distance <= 0:
            print("[WARN] SL/TP distance <= 0, пропускаем.")
            time.sleep(5)
            continue

        # размер позиции по риску
        volume = (balance * RISK_PER_TRADE) / sl_distance
        if volume <= 0:
            print("[WARN] volume <= 0, пропускаем.")
            time.sleep(5)
            continue

        # пересчёт SL/TP в цену
        point = symbol_info.point
        # Для синтетических индексов шаг цены уже в point, используем как есть
        if signal == "BUY":
            sl_price = price - sl_distance
            tp_price = price + tp_distance
            dir_int = 1
        else:
            sl_price = price + sl_distance
            tp_price = price - tp_distance
            dir_int = -1

        print(
            f"[ENTRY] signal={signal}, price={price:.5f}, SL={sl_price:.5f}, TP={tp_price:.5f}, "
            f"med_range={med_rng:.5f}, vol={volume:.4f}"
        )

        # отправляем ордер
        result = send_order(dir_int, volume, sl_price, tp_price, comment=f"AI_V25S_{RISK_MODE}")
        if result is not None:
            # логируем
            log_trade({
                "time": datetime.now(),
                "symbol": SYMBOL,
                "direction": signal,
                "price": price,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "volume": volume,
                "risk_mode": RISK_MODE,
                "dd_pct": dd_pct * 100,
                "day_loss_pct": day_loss_now_pct * 100,
                "prob_good": prob_good,
            })

        # ждём немного до следующей проверки
        time.sleep(5)


if __name__ == "__main__":
    try:
        run_agent()
    except KeyboardInterrupt:
        print("\n[AGENT] Остановлен пользователем (Ctrl+C).")
    finally:
        mt5.shutdown()
        print("[MT5] Выход.")
