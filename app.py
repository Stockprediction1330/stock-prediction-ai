# =============================================================================
# 【Streamlit版】株価予測AI - ローカルモデル版（市場休み対応・高速版）
# 43_【自作】ローカルモデル版_final.py をStreamlitアプリ化
# NaN自動回避システム + 保存済みモデルによる超高速予測
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import lightgbm as lgb
import yfinance as yf
from scipy import stats
import pickle
import os
import sqlite3
from datetime import datetime

# 警告を抑制
warnings.filterwarnings('ignore', category=FutureWarning)

# ページ設定
st.set_page_config(
    page_title="📈 株価予測AI（高速版）",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS - かっこいいデザイン
st.markdown("""
<style>
/* メインヘッダー */
.main-header {
    font-size: 3.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 900;
    text-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

/* サブヘッダー */
.sub-header {
    font-size: 1.2rem;
    color: #666;
    text-align: center;
    margin-bottom: 3rem;
    font-style: italic;
}

/* 予測カード */
.prediction-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    color: white;
    border: 1px solid rgba(255,255,255,0.2);
}

/* 指標カード */
.indicator-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 1rem;
    border-radius: 12px;
    margin: 0.5rem 0;
    box-shadow: 0 4px 16px rgba(240, 147, 251, 0.3);
    color: white;
    text-align: center;
}

/* データカード */
.data-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 1.2rem;
    border-radius: 12px;
    margin: 1rem 0;
    box-shadow: 0 6px 24px rgba(79, 172, 254, 0.3);
    color: white;
}

/* ネオンボタン効果 */
.neon-button {
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
    border: none;
    border-radius: 25px;
    padding: 12px 30px;
    color: white;
    font-weight: bold;
    box-shadow: 0 0 20px rgba(255, 107, 107, 0.5);
    transition: all 0.3s ease;
}

/* グラデーションテキスト */
.gradient-text {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: bold;
}

/* 指標テーブル */
.indicator-table {
    background: rgba(255,255,255,0.95);
    border-radius: 15px;
    padding: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.3);
}

/* アニメーション */
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(102, 126, 234, 0); }
    100% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0); }
}

.pulse-animation {
    animation: pulse 2s infinite;
}

/* ホログラム効果 */
.hologram {
    background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.5) 50%, transparent 70%);
    background-size: 200% 200%;
    animation: shimmer 3s infinite;
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

/* パルス効果 */
@keyframes glow {
    0%, 100% { 
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.5),
                   0 0 30px rgba(102, 126, 234, 0.3),
                   0 0 40px rgba(102, 126, 234, 0.1);
    }
    50% { 
        box-shadow: 0 0 30px rgba(102, 126, 234, 0.8),
                   0 0 40px rgba(102, 126, 234, 0.5),
                   0 0 50px rgba(102, 126, 234, 0.3);
    }
}

.glow-animation {
    animation: glow 3s ease-in-out infinite;
}

/* データカードホバー効果 */
.data-card:hover {
    transform: translateY(-2px);
    transition: all 0.3s ease;
    box-shadow: 0 8px 32px rgba(79, 172, 254, 0.5);
}

/* 成功インジケータ */
.success-pulse {
    animation: successPulse 2s ease-in-out infinite;
}

@keyframes successPulse {
    0%, 100% { 
        box-shadow: 0 0 20px rgba(34, 197, 94, 0.3);
        border-color: rgba(34, 197, 94, 0.6);
    }
    50% { 
        box-shadow: 0 0 30px rgba(34, 197, 94, 0.6);
        border-color: rgba(34, 197, 94, 1);
    }
}

/* レスポンシブ */
@media (max-width: 768px) {
    .main-header { font-size: 2.5rem; }
}
</style>
""", unsafe_allow_html=True)

def check_password():
    """パスワード認証"""
    def password_entered():
        if st.session_state["password"] == "stock2024":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # パスワードを削除
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # 初回アクセス
        st.markdown('<div class="main-header">🔐 株価予測AI - ログイン</div>', unsafe_allow_html=True)
        st.text_input(
            "パスワードを入力してください", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.info("📝 アクセスコードが必要です")
        return False
    elif not st.session_state["password_correct"]:
        # パスワード間違い
        st.markdown('<div class="main-header">🔐 株価予測AI - ログイン</div>', unsafe_allow_html=True)
        st.text_input(
            "パスワードを入力してください", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("❌ パスワードが正しくありません")
        return False
    else:
        # 認証成功
        return True

@st.cache_data(ttl=3600)  # 1時間キャッシュ
def load_saved_models():
    """保存済みモデルと特徴量リストを読み込み"""
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        st.error("❌ modelsフォルダが見つかりません。")
        return None, None, None, None
    
    try:
        # 回帰モデル読み込み
        with open(f"{models_dir}/regression_model.pkl", "rb") as f:
            regression_model = pickle.load(f)
        
        # 分類モデル読み込み
        with open(f"{models_dir}/classification_model.pkl", "rb") as f:
            classification_model = pickle.load(f)
        
        # 特徴量リスト読み込み
        with open(f"{models_dir}/features_regression.pkl", "rb") as f:
            features_forREG = pickle.load(f)
        
        with open(f"{models_dir}/features_classification.pkl", "rb") as f:
            features_forCLASS = pickle.load(f)
        
        return regression_model, classification_model, features_forREG, features_forCLASS
        
    except Exception as e:
        st.error(f"❌ モデル読み込みエラー: {e}")
        return None, None, None, None

def find_latest_valid_data(df, column_name):
    """最新の有効なデータ（NaN以外）のインデックスと値を取得"""
    if column_name not in df.columns:
        return None, None, None
    
    # 後ろから順番に有効なデータを探す
    for i in range(len(df) - 1, -1, -1):
        value = df.iloc[i][column_name]
        if pd.notna(value) and value != 0:  # NaNまたは0以外
            date = df.iloc[i]['Date']
            return i, value, date
    
    return None, None, None

@st.cache_data(ttl=300)  # 5分キャッシュ
def download_current_stock_data(ticker_tgt="1330.T", period="100d"):
    """対象銘柄の現在の株価データをダウンロードし、NaN値を自動的に回避"""
    df_raw = yf.download(ticker_tgt, period=period, progress=False)
    
    if df_raw.empty:
        return None
    
    # データの前処理
    df_tgt = pd.DataFrame(df_raw)
    df_tgt = df_tgt.reset_index()
    df_tgt['date'] = pd.to_datetime(df_tgt['Date'])
    df_tgt.drop(columns=['Date'], inplace=True)
    df_tgt.drop(columns=['High', 'Low'], inplace=True)
    
    # 一貫性のための列名変更
    new_columns = ['close', 'open', 'volume', 'date']
    df_tgt = pd.DataFrame(df_tgt.values, columns=new_columns)
    
    # 移動平均の計算
    df_tgt['SMA_5'] = df_tgt['close'].rolling(window=5).mean()
    df_tgt['SMA_25'] = df_tgt['close'].rolling(window=25).mean()
    df_tgt['SMA_50'] = df_tgt['close'].rolling(window=50).mean()
    
    # 移動平均からの乖離率を計算
    df_tgt['SMA_5_ratio'] = (df_tgt['close'] - df_tgt['SMA_5']) / df_tgt['SMA_5'] * 100
    df_tgt['SMA_25_ratio'] = (df_tgt['close'] - df_tgt['SMA_25']) / df_tgt['SMA_25'] * 100
    df_tgt['SMA_50_ratio'] = (df_tgt['close'] - df_tgt['SMA_50']) / df_tgt['SMA_50'] * 100
    
    # 移動平均間の相対比率を計算
    df_tgt['SMA_5_25_ratio'] = df_tgt['SMA_25_ratio'] - df_tgt['SMA_5_ratio']
    df_tgt['SMA_5_50_ratio'] = df_tgt['SMA_50_ratio'] - df_tgt['SMA_5_ratio']
    
    return df_tgt

@st.cache_data(ttl=300)  # 5分キャッシュ
def download_market_indicators(period="100d"):
    """様々な市場指標データをダウンロードし、NaN値を自動的に回避"""
    # 追跡する市場指標の定義
    tickers = {
        "Dow_Jones": "^DJI",
        "SP500": "^GSPC", 
        "Nasdaq": "^IXIC",
        "WTI_Crude": "CL=F", 
        "VIX": "^VIX",
        "USD_JPY": "USDJPY=X",
        "US_2y_yield": "^FVX",
        "US_10y_yield": "^TNX",
        "Gold_future": "GC=F",
        "CME_Nikkei_future": "NKD=F"
    }
    
    # データ保存用
    data = {}
    
    # 各指標のデータをダウンロード
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, period=period, progress=False)
            
            if df.empty:
                continue
                
            close_series = df["Close"]
            close_df = close_series.reset_index()
            close_df.columns = ["Date", name]
            data[name] = close_df
                
        except Exception as e:
            continue
    
    if not data:
        return None
    
    # データの結合
    result_df = data[list(data.keys())[0]].copy()
    for name in list(data.keys())[1:]:
        result_df = pd.merge(result_df, data[name], on="Date", how="outer")
    
    # 日付でソート
    result_df = result_df.sort_values(by="Date")
    
    # NaN回避対応の変化率計算
    df_combined_pct = calculate_pct_change_with_nan_handling(result_df)
    
    return df_combined_pct

def calculate_pct_change_with_nan_handling(df):
    """NaN値を回避して変化率を計算する"""
    df_result = df.copy()
    
    # Date列以外の全ての列に対して処理
    for column in df.columns:
        if column == 'Date':
            continue
            
        # 各行に対して変化率を計算
        for i in range(len(df)):
            if i == 0:
                # 最初の行は0%
                df_result.iloc[i, df_result.columns.get_loc(column)] = 0.0
                continue
                
            # 現在行から最新の有効値を探す
            current_value = None
            current_date = None
            for curr_idx in range(i, -1, -1):  # 現在行から逆順で探索
                val = df.iloc[curr_idx][column]
                if pd.notna(val) and val != 0:
                    current_value = val
                    current_date = df.iloc[curr_idx]['Date']
                    break
            
            # 前営業日の有効値を探す
            prev_valid_value = None
            prev_date = None
            days_back = 0
            
            if current_value is not None:
                # 現在の有効値が見つかった地点より前から探索
                search_start = curr_idx - 1 if 'curr_idx' in locals() else i - 1
                for j in range(search_start, -1, -1):
                    prev_val = df.iloc[j][column]
                    if pd.notna(prev_val) and prev_val != 0:
                        prev_valid_value = prev_val
                        prev_date = df.iloc[j]['Date']
                        days_back = curr_idx - j
                        break
            
            # 変化率計算
            if current_value is not None and prev_valid_value is not None and prev_valid_value != 0:
                pct_change = ((current_value - prev_valid_value) / prev_valid_value) * 100
                df_result.iloc[i, df_result.columns.get_loc(column)] = pct_change
            else:
                # 有効値が見つからない場合は0%
                df_result.iloc[i, df_result.columns.get_loc(column)] = 0.0
    
    return df_result

def combine_stock_and_market_data(df_stock, df_market):
    """株価データと市場指標を結合し、派生特徴量を計算する"""
    # データフレームの結合
    combined_data = pd.merge(df_market, df_stock, left_on='Date', right_on='date', how='inner')
    combined_data.drop(columns=['date', 'SMA_5', 'SMA_25', 'SMA_50'], inplace=True)
    
    # 前日の特徴量を計算
    combined_data['prevday_open_close_ratio'] = (combined_data['close'] - combined_data['open']) / combined_data['open'] * 100
    combined_data['prevday_open_close_ratio'] = combined_data['prevday_open_close_ratio'].fillna(combined_data['prevday_open_close_ratio'].median())
    
    combined_data['prevday_close_ratio'] = (combined_data['close'] - combined_data['close'].shift(1)) / combined_data['close'].shift(1) * 100
    combined_data['prevday_close_ratio'] = combined_data['prevday_close_ratio'].fillna(combined_data['prevday_close_ratio'].median())
    
    combined_data['prevday_volume_ratio'] = (combined_data['volume'] - combined_data['volume'].shift(1)) / combined_data['volume'].shift(1) * 100
    combined_data['prevday_volume_ratio'] = combined_data['prevday_volume_ratio'].fillna(combined_data['prevday_volume_ratio'].median())
    
    # 0値を前日値で補完（米国債利回りの市場開始前対応）
    for col in ['US_10y_yield', 'US_2y_yield']:
        if col in combined_data.columns:
            # 最新行が0値の場合、前日の有効値で補完
            if combined_data[col].iloc[-1] == 0.0:
                # 前の行から有効値を探す
                for i in range(len(combined_data)-2, -1, -1):
                    if combined_data[col].iloc[i] != 0.0 and not pd.isna(combined_data[col].iloc[i]):
                        combined_data.loc[combined_data.index[-1], col] = combined_data[col].iloc[i]
                        break

    # イールドスプレッドの計算
    combined_data['10_2_yield'] = combined_data['US_10y_yield'] - combined_data['US_2y_yield']
    
    # イールドスプレッドの変化を計算
    data_was_compensated = False
    if len(combined_data) >= 2:
        latest_yield = combined_data['10_2_yield'].iloc[-1]
        prev_yield = combined_data['10_2_yield'].iloc[-2]
        if latest_yield == prev_yield and latest_yield != 0:
            data_was_compensated = True
    
    if data_was_compensated and len(combined_data) >= 2:
        # 0値補完があった場合：有効な比較対象を見つけるまで遡る
        current_yield = combined_data['10_2_yield'].iloc[-2]  # 補完に使った前日値
        comparison_yield = None
        
        # 前々日から遡って有効な値を探す
        for i in range(len(combined_data)-3, -1, -1):
            candidate_yield = combined_data['10_2_yield'].iloc[i]
            if candidate_yield != 0.0 and not pd.isna(candidate_yield):
                comparison_yield = candidate_yield
                break
        
        if comparison_yield is not None:
            yield_change = current_yield - comparison_yield
            # 最新行に設定（通常の計算も実行してから上書き）
            combined_data['10_2_yield_change'] = combined_data['10_2_yield'] - combined_data['10_2_yield'].shift(1)
            combined_data.loc[combined_data.index[-1], '10_2_yield_change'] = yield_change
        else:
            # 有効な比較対象が見つからない場合は通常処理
            combined_data['10_2_yield_change'] = combined_data['10_2_yield'] - combined_data['10_2_yield'].shift(1)
    else:
        # 通常の場合：最新値 - 前日値
        combined_data['10_2_yield_change'] = combined_data['10_2_yield'] - combined_data['10_2_yield'].shift(1)
    
    # 前日の値を保存
    combined_data['prevday_USD_JPY'] = combined_data['USD_JPY']
    combined_data['prevday_USD_JPY'] = combined_data['prevday_USD_JPY'].ffill()
    
    combined_data['prevday_WTI_Crude'] = combined_data['WTI_Crude'] 
    combined_data['prevday_WTI_Crude'] = combined_data['prevday_WTI_Crude'].ffill()
    
    combined_data['prevday_Gold_future'] = combined_data['Gold_future']
    combined_data['prevday_Gold_future'] = combined_data['prevday_Gold_future'].ffill()
    
    # 合成特徴量の計算
    if 'SP500' in combined_data.columns and 'Dow_Jones' in combined_data.columns:
        combined_data['US_market_composite'] = (combined_data['SP500'] + combined_data['Dow_Jones']) / 2
        combined_data['US_stocks_volatility'] = abs(combined_data['SP500'] - combined_data['Dow_Jones'])
    
    if 'WTI_Crude' in combined_data.columns and 'USD_JPY' in combined_data.columns:
        combined_data['oil_currency_ratio'] = combined_data['WTI_Crude'] / combined_data['USD_JPY']
    
    # 残りの欠損値をカラムの中央値で埋める
    for col in combined_data.columns:
        if col != 'Date' and combined_data[col].isnull().any():
            col_median = combined_data[col].median()
            combined_data[col] = combined_data[col].fillna(col_median)
    
    return combined_data

def detect_market_regime(data, lookback_period=60):
    """価格トレンドとボラティリティに基づいて市場レジームを検出する"""
    # ルックバック期間の最近のデータを取得
    recent_data = data.sort_values('Date').tail(lookback_period)
    
    # 年率換算ボラティリティを計算
    price_vol = recent_data['close'].pct_change().std() * np.sqrt(252)
    
    # 過去データと比較したボラティリティのパーセンタイルを計算
    all_rolling_vol = data['close'].pct_change().rolling(20).std() * np.sqrt(252)
    vol_percentile = pd.Series(all_rolling_vol).rank(pct=True).iloc[-1]
    
    # 価格トレンドを検出
    start_price = recent_data['close'].iloc[0]
    end_price = recent_data['close'].iloc[-1]
    price_change = (end_price / start_price - 1) * 100
    price_direction = np.sign(price_change)
    
    # 線形回帰を使用してトレンドの強さを計算
    x = np.arange(len(recent_data))
    y = recent_data['close'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # トレンド強度インジケータを計算
    trend_strength = r_value ** 2 * price_direction
    
    # トレンドレジームの分類
    if trend_strength > 0.6:
        trend_regime = 'strong_uptrend'
    elif trend_strength > 0.3:
        trend_regime = 'uptrend'
    elif trend_strength < -0.6:
        trend_regime = 'strong_downtrend'
    elif trend_strength < -0.3:
        trend_regime = 'downtrend'
    else:
        trend_regime = 'range_bound'
    
    # ボラティリティレジームの分類
    if vol_percentile > 0.7:
        vol_regime = 'high_volatility'
    elif vol_percentile < 0.3:
        vol_regime = 'low_volatility'
    else:
        vol_regime = 'normal_volatility'
    
    return {
        'trend_regime': trend_regime,
        'volatility_regime': vol_regime,
        'trend_strength': trend_strength,
        'volatility': price_vol,
        'vol_percentile': vol_percentile,
        'price_change': price_change
    }

def add_regime_features(data, lookback_period=60):
    """データセットに市場レジーム特徴量を追加する"""
    enhanced_data = data.copy()
    
    # 各時点で十分な履歴を持つレジーム特徴量を計算
    for i in range(lookback_period, len(data)):
        window_data = data.iloc[i-lookback_period:i]
        
        try:
            regime_info = detect_market_regime(window_data)
            
            # 数値特徴量の追加
            enhanced_data.loc[data.index[i], 'trend_strength'] = regime_info['trend_strength']
            enhanced_data.loc[data.index[i], 'volatility_percentile'] = regime_info['vol_percentile']
            
            # カテゴリ特徴量を数値にエンコード
            trend_map = {
                'strong_uptrend': 2, 
                'uptrend': 1, 
                'range_bound': 0, 
                'downtrend': -1, 
                'strong_downtrend': -2
            }
            
            vol_map = {
                'high_volatility': 2, 
                'normal_volatility': 1, 
                'low_volatility': 0
            }
            
            enhanced_data.loc[data.index[i], 'trend_regime_numeric'] = trend_map[regime_info['trend_regime']]
            enhanced_data.loc[data.index[i], 'vol_regime_numeric'] = vol_map[regime_info['volatility_regime']]
            
        except Exception as e:
            pass
    
    # レジーム列のNaN値を前方に埋める
    regime_cols = ['trend_strength', 'volatility_percentile', 'trend_regime_numeric', 'vol_regime_numeric']
    for col in regime_cols:
        enhanced_data[col] = enhanced_data[col].ffill()
    
    return enhanced_data

def predict_today(todays_data, regression_model, classification_model, features_forREG, features_forCLASS):
    """今日のデータに対する予測を生成"""
    # 今日の特徴量を準備
    X_today = pd.DataFrame(todays_data, columns=todays_data.columns, index=todays_data.index)
    X_today_forREG = X_today[features_forREG]
    X_today_forCLASS = X_today[features_forCLASS]
    
    # 予測を生成
    y_today_pred = regression_model.predict(X_today_forREG)
    y_today_pred_updown_proba = classification_model.predict_proba(X_today_forCLASS)[:, 1]
    
    # 取引シグナルの判定
    if y_today_pred[0] > 0 and y_today_pred_updown_proba[0] >= 0.5:
        trading_signal = "買い"
    elif y_today_pred[0] < 0 and y_today_pred_updown_proba[0] <= 0.5:
        trading_signal = "売り"
    else:
        trading_signal = "様子見"
    
    return {
        "regression": y_today_pred[0],
        "classification": y_today_pred_updown_proba[0] * 100,
        "signal": trading_signal
    }

def main_app():
    """メインアプリケーション"""
    st.markdown('<div class="main-header">📈 株価予測AI（高速版）</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">🚀 NaN自動回避システム + 保存済みモデルによる超高速予測</div>', unsafe_allow_html=True)
    
    # サイドバー - かっこいいデザイン
    st.sidebar.markdown('<div class="gradient-text" style="font-size: 1.5rem;">⚙️ システム情報</div>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("""
    <div class="indicator-card">
        <h4>📊 対象銘柄</h4>
        <p><strong>1330</strong><br>MAXIS 海外株式</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    <div class="indicator-card">
        <h4>🤖 AIモデル</h4>
        <p><strong>LightGBM</strong><br>回帰+分類 ハイブリッド</p>
    </div>
    """, unsafe_allow_html=True)
    
    current_time = datetime.now().strftime('%H:%M:%S')
    st.sidebar.markdown(f"""
    <div class="indicator-card glow-animation">
        <h4>🔄 データ更新</h4>
        <p><strong>リアルタイム</strong><br>Yahoo Finance API</p>
        <p style="font-size: 0.8rem; opacity: 0.8;">最終更新: {current_time}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # モデル読み込み
    with st.spinner('🔄 モデル読み込み中...'):
        regression_model, classification_model, features_forREG, features_forCLASS = load_saved_models()
    
    if regression_model is None:
        st.error("❌ モデルの読み込みに失敗しました。")
        st.info("💡 43番のローカルモデル版を先に実行してモデルを保存してください。")
        return
    
    st.success(f"✅ モデル読み込み完了 (回帰: {len(features_forREG)}特徴量, 分類: {len(features_forCLASS)}特徴量)")
    
    # リアルタイム指標表示
    st.markdown('<div class="gradient-text" style="font-size: 1.8rem; text-align: center;">📊 リアルタイム市場指標</div>', unsafe_allow_html=True)
    
    # リアルタイム指標データを取得
    with st.spinner('📊 市場指標取得中...'):
        try:
            df_market_realtime = download_market_indicators("5d")
            df_stock_realtime = download_current_stock_data("1330.T", "30d")
            
            if df_market_realtime is not None and df_stock_realtime is not None:
                # 最新の市場指標データを抽出
                latest_market = df_market_realtime.iloc[-1]
                latest_stock = df_stock_realtime.iloc[-1]
                
                market_indicators = {
                    "米国株指標": {},
                    "債券・通貨": {},
                    "日本株": {}
                }
                
                # シンプルに既存データから取得（Yahoo Finance APIは複雑なので）
                try:
                    # 個別にデータ取得して終値を表示
                    symbols = {"^DJI": "ダウ平均", "^GSPC": "S&P500", "^IXIC": "ナスダック", "^VIX": "VIX"}
                    for symbol, name in symbols.items():
                        try:
                            ticker_data = yf.download(symbol, period="2d", progress=False)
                            if not ticker_data.empty and len(ticker_data) >= 2:
                                current_price = float(ticker_data['Close'].iloc[-1])
                                prev_price = float(ticker_data['Close'].iloc[-2])
                                change_pct = ((current_price - prev_price) / prev_price) * 100
                                
                                # VIXは上昇が悪材料なので色を反転
                                if symbol == "^VIX":
                                    color = "🔴" if change_pct >= 0 else "🟢"
                                else:
                                    color = "🟢" if change_pct >= 0 else "🔴"
                                
                                market_indicators["米国株指標"][name] = {
                                    "value": current_price,
                                    "change": change_pct,
                                    "color": color
                                }
                        except:
                            continue
                    
                    # 債券・通貨指標（終値表示）
                    # ドル円
                    try:
                        usd_jpy_data = yf.download("USDJPY=X", period="2d", progress=False)
                        if not usd_jpy_data.empty and len(usd_jpy_data) >= 2:
                            current_usd_jpy = float(usd_jpy_data['Close'].iloc[-1])
                            prev_usd_jpy = float(usd_jpy_data['Close'].iloc[-2])
                            usd_jpy_change_pct = ((current_usd_jpy - prev_usd_jpy) / prev_usd_jpy) * 100
                            
                            market_indicators["債券・通貨"]["ドル円"] = {
                                "value": current_usd_jpy,
                                "change": usd_jpy_change_pct,
                                "color": "🟢" if usd_jpy_change_pct >= 0 else "🔴"
                            }
                    except:
                        pass
                    
                    # 原油
                    try:
                        oil_data = yf.download("CL=F", period="2d", progress=False)
                        if not oil_data.empty and len(oil_data) >= 2:
                            current_oil = float(oil_data['Close'].iloc[-1])
                            prev_oil = float(oil_data['Close'].iloc[-2])
                            oil_change_pct = ((current_oil - prev_oil) / prev_oil) * 100
                            
                            market_indicators["債券・通貨"]["原油"] = {
                                "value": current_oil,
                                "change": oil_change_pct,
                                "color": "🟢" if oil_change_pct >= 0 else "🔴"
                            }
                    except:
                        pass
                    
                    # 日本株（1330の終値）
                    try:
                        japan_stock = yf.download("1330.T", period="2d", progress=False)
                        if not japan_stock.empty and len(japan_stock) >= 2:
                            current_price = float(japan_stock['Close'].iloc[-1])
                            prev_price = float(japan_stock['Close'].iloc[-2])
                            change_pct = ((current_price - prev_price) / prev_price) * 100
                            
                            market_indicators["日本株"]["1330終値"] = {
                                "value": current_price,
                                "change": change_pct,
                                "color": "🟢" if change_pct >= 0 else "🔴"
                            }
                    except:
                        pass
                        
                except Exception as inner_e:
                    st.warning(f"一部データの取得に失敗: {str(inner_e)[:50]}")
            else:
                # データ取得に失敗した場合のフォールバック
                market_indicators = {
                    "米国株指標": {
                        "ダウ平均": {"value": 0.0, "change": 0.0, "color": "⚪"},
                        "S&P500": {"value": 0.0, "change": 0.0, "color": "⚪"},
                        "ナスダック": {"value": 0.0, "change": 0.0, "color": "⚪"},
                        "VIX": {"value": 0.0, "change": 0.0, "color": "⚪"}
                    },
                    "債券・通貨": {
                        "ドル円": {"value": 0.0, "change": 0.0, "color": "⚪"},
                        "原油": {"value": 0.0, "change": 0.0, "color": "⚪"}
                    },
                    "日本株": {
                        "1330終値": {"value": 0.0, "change": 0.0, "color": "⚪"}
                    }
                }
                st.warning("⚠️ 市場指標の取得に時間がかかっています")
        except Exception as e:
            # エラー時のフォールバック
            market_indicators = {
                "エラー": {
                    "データ取得": {"value": 0.0, "change": 0.0, "color": "❌"}
                }
            }
            st.error(f"❌ 市場指標取得エラー: {str(e)[:100]}...")
    
    # 3列に分けて表示
    col1, col2, col3 = st.columns(3)
    
    categories = list(market_indicators.keys())
    columns = [col1, col2, col3]
    
    for i, (category, indicators) in enumerate(market_indicators.items()):
        with columns[i]:
            st.markdown(f"""
            <div class="data-card">
                <h3 style="margin-top: 0;">{category}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for name, data in indicators.items():
                color = data["color"]
                value = data["value"]
                change = data.get("change", 0.0)
                
                # 終値表示の場合は絶対値、変化率表示の場合はパーセント表示
                if "終値" in name or name in ["ダウ平均", "S&P500", "ナスダック", "VIX", "ドル円", "原油"]:
                    if name == "ドル円":
                        value_display = f"{value:.2f}円"
                    elif name == "原油":
                        value_display = f"${value:.2f}"
                    else:
                        value_display = f"{value:.0f}"
                    change_display = f"({change:+.2f}%)" if change != 0 else ""
                else:
                    value_display = f"{value:+.2f}%"
                    change_display = ""
                
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.1); padding: 0.5rem; margin: 0.3rem 0; border-radius: 8px; display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 500;">{name}</span>
                    <span style="font-weight: bold;">{color} {value_display} <small>{change_display}</small></span>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 予測実行ボタン
    if st.button("🔮 株価予測を実行", type="primary", use_container_width=True):
        with st.spinner('📊 データ取得・予測実行中...'):
            try:
                # 注意メッセージ
                st.info("⚡ **超高速モード**: 保存済みモデルを使用して予測中...")
                
                # リアルタイムデータ取得
                df_tgt = download_current_stock_data("1330.T")
                if df_tgt is None:
                    st.error("❌ 株価データの取得に失敗しました")
                    return
                
                df_combined = download_market_indicators()
                if df_combined is None:
                    st.error("❌ 市場指標データの取得に失敗しました")
                    return
                
                # データ結合と特徴量計算
                todays_data = combine_stock_and_market_data(df_tgt, df_combined)
                if todays_data is None or todays_data.empty:
                    st.error("❌ データの結合に失敗しました")
                    return
                
                # すべての列を数値に変換
                for col in todays_data.columns:
                    if col != 'Date':
                        try:
                            todays_data[col] = pd.to_numeric(todays_data[col], errors='coerce')
                        except Exception as e:
                            pass
                
                # 市場レジーム特徴量の追加
                todays_data = add_regime_features(todays_data)
                
                # テストデータの準備
                drop_cols = ['Date', 'open', 'close', 'volume', 'USD_JPY', 'WTI_Crude',
                            'Gold_future', 'CME_Nikkei_future', 
                            'prevday_WTI_Crude', 'prevday_Gold_future', 'prevday_close_ratio']
                
                todays_test_data = todays_data.drop(columns=drop_cols)
                todays_test_data = todays_test_data.tail(1)
                
                # 実際の予測実行
                prediction_result = predict_today(
                    todays_test_data, regression_model, classification_model, 
                    features_forREG, features_forCLASS
                )
                
                st.success("🎉 予測完了！")
                
                # 結果表示
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="prediction-card pulse-animation">', unsafe_allow_html=True)
                    st.metric(
                        label="📈 回帰モデル予測", 
                        value=f"{prediction_result['regression']:.3f}%",
                        delta="始値→終値の変化率",
                        help="LightGBM回帰モデルによる予測"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="prediction-card pulse-animation">', unsafe_allow_html=True)
                    st.metric(
                        label="🎯 分類モデル予測", 
                        value=f"{prediction_result['classification']:.1f}%",
                        delta="上昇確率",
                        help="LightGBM分類モデルによる上昇確率"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    signal_card_class = "prediction-card success-pulse" if prediction_result['signal'] != "様子見" else "prediction-card pulse-animation"
                    st.markdown(f'<div class="{signal_card_class}">', unsafe_allow_html=True)
                    signal_color = "🔴" if prediction_result['signal'] == "売り" else "🟢" if prediction_result['signal'] == "買い" else "🟡"
                    st.metric(
                        label="💡 取引シグナル", 
                        value=f"{signal_color} {prediction_result['signal']}",
                        delta="ハイブリッド判定",
                        help="回帰+分類モデルの統合判定"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # 詳細情報
                st.markdown("### 📋 予測詳細")
                
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.markdown("**🔢 数値結果**")
                    st.markdown(f"• 回帰予測: **{prediction_result['regression']:.4f}%**")
                    st.markdown(f"• 上昇確率: **{prediction_result['classification']:.2f}%**")
                    st.markdown(f"• 取引推奨: **{prediction_result['signal']}**")
                
                with detail_col2:
                    st.markdown("**⚙️ システム情報**")
                    st.markdown("• **NaN自動回避**: ✅ 動作中")
                    st.markdown("• **市場休み対応**: ✅ 対応済み")
                    st.markdown("• **モデル**: 保存済み高速版")
                    st.markdown(f"• **データ更新**: {datetime.now().strftime('%H:%M:%S')}**")
                
            except Exception as e:
                st.error(f"❌ 予測エラー: {e}")
                st.info("💡 43番のローカルモデル版が正常に動作することを確認してください。")
    
    # フッター
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        🤖 Generated with Claude Code | 📊 Powered by LightGBM
        </div>
        """, 
        unsafe_allow_html=True
    )

def main():
    """メイン関数"""
    if check_password():
        main_app()

if __name__ == "__main__":
    main()