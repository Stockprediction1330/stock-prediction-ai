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

# 警告を抑制
warnings.filterwarnings('ignore', category=FutureWarning)

# ページ設定
st.set_page_config(
    page_title="📈 株価予測AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-result {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
}
.success-box {
    background-color: #d4edda;
    color: #155724;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #c3e6cb;
}
.warning-box {
    background-color: #fff3cd;
    color: #856404;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #ffeaa7;
}
.error-box {
    background-color: #f8d7da;
    color: #721c24;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #f5c6cb;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# パスワード認証
# =============================================================================

def check_password():
    """パスワード認証機能"""
    
    # セッション状態の初期化
    if 'password_correct' not in st.session_state:
        st.session_state.password_correct = False
    
    # 既に認証済みの場合はTrueを返す
    if st.session_state.password_correct:
        return True
    
    # パスワード入力UI
    st.markdown('<div class="main-header">🔐 株価予測AI - ログイン</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🔑 アクセス認証が必要です
    
    このアプリケーションは認証が必要です。  
    正しいパスワードを入力してください。
    """)
    
    # パスワード入力
    password = st.text_input("パスワードを入力", type="password", key="password_input")
    
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        login_button = st.button("🔓 ログイン", use_container_width=True)
    
    # ログインボタンまたはEnterキーでパスワード確認
    if login_button or password:
        if password == "stock2024":  # 正しいパスワード
            st.session_state.password_correct = True
            st.success("✅ 認証成功！アプリにアクセスしています...")
            st.rerun()
        elif password:  # パスワードが入力されているが間違っている
            st.error("❌ パスワードが間違っています")
    
    # フッター情報
    st.markdown("""
    ---
    **📝 備考:**  
    - パスワードは友人から共有されたものを使用してください  
    - 認証後は自動的にメイン画面に移動します
    """)
    
    return False

# =============================================================================
# モデル読み込み関数
# =============================================================================

@st.cache_data(ttl=3600)  # 1時間キャッシュ
def load_saved_models():
    """保存済みモデルと特徴量リストを読み込み"""
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        st.error("❌ modelsフォルダが見つかりません。先に42_【自作】指標取り込み精度向上版.pyを実行してモデルを保存してください。")
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
            features_reg = pickle.load(f)
            
        with open(f"{models_dir}/features_classification.pkl", "rb") as f:
            features_cls = pickle.load(f)
        
        return regression_model, classification_model, features_reg, features_cls
        
    except Exception as e:
        st.error(f"❌ モデル読み込みエラー: {e}")
        return None, None, None, None

# =============================================================================
# データ取得・処理関数（43番から完全コピー）
# =============================================================================

@st.cache_data(ttl=300)  # 5分キャッシュ
def download_current_stock_data(ticker_tgt="1330.T", period="100d", market_holiday=None):
    """対象銘柄の現在の株価データをダウンロードして前処理する"""
    try:
        df_tgt = yf.download(ticker_tgt, period=period)
        
        # データの前処理
        df_tgt = pd.DataFrame(df_tgt)
        df_tgt = df_tgt.reset_index()
        df_tgt['date'] = pd.to_datetime(df_tgt['Date'])
        
        # 市場休みに基づく日付調整
        if market_holiday == 'japan':
            df_tgt['date'] += pd.Timedelta(days=1)
        elif market_holiday == 'us':
            df_tgt['date'] -= pd.Timedelta(days=1)
        
        df_tgt.drop(columns=['Date'], inplace=True)
        df_tgt.drop(columns=['High', 'Low'], inplace=True)
        
        # 列名変更
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
        
    except Exception as e:
        st.error(f"株価データ取得エラー: {e}")
        return None

@st.cache_data(ttl=300)  # 5分キャッシュ
def download_market_indicators(period="100d"):
    """市場指標データをダウンロードする"""
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
    
    try:
        data = {}
        for name, ticker in tickers.items():
            try:
                df = yf.download(ticker, period=period)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                close_series = df["Close"]
                close_df = close_series.reset_index()
                close_df.columns = ["Date", name]
                data[name] = close_df
            except:
                continue
        
        # データ結合
        result_df = data[list(tickers.keys())[0]].copy()
        for name in list(tickers.keys())[1:]:
            if name in data:
                result_df = pd.merge(result_df, data[name], on="Date", how="outer")
        
        result_df = result_df.sort_values(by="Date")
        
        # 変化率計算
        df_combined = result_df.copy()
        
        # SP500とDow_Jonesの手動変化率計算
        sp500_change = None
        dow_change = None
        
        for current_idx in range(len(result_df)-1, max(-1, len(result_df)-5), -1):
            current_row = result_df.iloc[current_idx]
            current_sp = current_row.get('SP500', None)
            current_dow = current_row.get('Dow_Jones', None)
            
            if pd.notna(current_sp) and pd.notna(current_dow):
                for prev_idx in range(current_idx-1, max(-1, current_idx-5), -1):
                    prev_row = result_df.iloc[prev_idx]
                    prev_sp = prev_row.get('SP500', None)
                    prev_dow = prev_row.get('Dow_Jones', None)
                    
                    if pd.notna(prev_sp) and pd.notna(prev_dow) and prev_sp > 0 and prev_dow > 0:
                        sp500_change = ((current_sp - prev_sp) / prev_sp) * 100
                        dow_change = ((current_dow - prev_dow) / prev_dow) * 100
                        break
                
                if sp500_change is not None:
                    break
        
        # 通常のpct_change計算
        df_combined.iloc[:, 1:] = df_combined.iloc[:, 1:].pct_change() * 100
        
        # 手動計算した値で上書き
        if sp500_change is not None:
            df_combined.iloc[-1, df_combined.columns.get_loc('SP500')] = sp500_change
        if dow_change is not None:
            df_combined.iloc[-1, df_combined.columns.get_loc('Dow_Jones')] = dow_change
        
        return df_combined
        
    except Exception as e:
        st.error(f"市場指標データ取得エラー: {e}")
        return None

def combine_stock_and_market_data(df_stock, df_market):
    """株価データと市場指標を結合し、派生特徴量を計算する"""
    try:
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
        
        # イールドスプレッドの計算
        combined_data['10_2_yield'] = combined_data['US_10y_yield'] - combined_data['US_2y_yield']
        
        # 前日の値を保存
        combined_data['prevday_USD_JPY'] = combined_data['USD_JPY']
        combined_data['prevday_USD_JPY'] = combined_data['prevday_USD_JPY'].ffill()
        
        combined_data['prevday_WTI_Crude'] = combined_data['WTI_Crude'] 
        combined_data['prevday_WTI_Crude'] = combined_data['prevday_WTI_Crude'].ffill()
        
        combined_data['prevday_Gold_future'] = combined_data['Gold_future']
        combined_data['prevday_Gold_future'] = combined_data['prevday_Gold_future'].ffill()
        
        # 合成特徴量の計算
        if 'SP500' in combined_data.columns and 'Dow_Jones' in combined_data.columns:
            sp500_clean = combined_data['SP500'].replace(0, np.nan)
            dow_clean = combined_data['Dow_Jones'].replace(0, np.nan)
            
            combined_data['US_market_composite'] = (sp500_clean + dow_clean) / 2
            combined_data['US_stocks_volatility'] = abs(sp500_clean - dow_clean)
            
            combined_data['US_market_composite'] = combined_data['US_market_composite'].ffill()
            combined_data['US_stocks_volatility'] = combined_data['US_stocks_volatility'].ffill()
        
        if 'WTI_Crude' in combined_data.columns and 'USD_JPY' in combined_data.columns:
            combined_data['oil_currency_ratio'] = combined_data['WTI_Crude'] / combined_data['USD_JPY']
        
        # イールドスプレッドの変化を計算
        combined_data['10_2_yield_change'] = combined_data['10_2_yield'] - combined_data['10_2_yield'].shift(1)
        
        # 残りの欠損値をカラムの中央値で埋める
        for col in combined_data.columns:
            if col != 'Date' and combined_data[col].isnull().any():
                col_median = combined_data[col].median()
                combined_data[col] = combined_data[col].fillna(col_median)
        
        return combined_data
        
    except Exception as e:
        st.error(f"データ結合エラー: {e}")
        return None

def detect_market_regime(data, lookback_period=60):
    """価格トレンドとボラティリティに基づいて市場レジームを検出する"""
    try:
        recent_data = data.sort_values('Date').tail(lookback_period)
        
        price_vol = recent_data['close'].pct_change().std() * np.sqrt(252)
        all_rolling_vol = data['close'].pct_change().rolling(20).std() * np.sqrt(252)
        vol_percentile = pd.Series(all_rolling_vol).rank(pct=True).iloc[-1]
        
        start_price = recent_data['close'].iloc[0]
        end_price = recent_data['close'].iloc[-1]
        price_change = (end_price / start_price - 1) * 100
        price_direction = np.sign(price_change)
        
        x = np.arange(len(recent_data))
        y = recent_data['close'].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        trend_strength = r_value ** 2 * price_direction
        
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
    except:
        return {
            'trend_regime': 'range_bound',
            'volatility_regime': 'normal_volatility',
            'trend_strength': 0.0,
            'volatility': 0.2,
            'vol_percentile': 0.5,
            'price_change': 0.0
        }

def add_regime_features(data, lookback_period=60):
    """データセットに市場レジーム特徴量を追加する"""
    enhanced_data = data.copy()
    
    for i in range(lookback_period, len(data)):
        window_data = data.iloc[i-lookback_period:i]
        
        try:
            regime_info = detect_market_regime(window_data)
            
            enhanced_data.loc[data.index[i], 'trend_strength'] = regime_info['trend_strength']
            enhanced_data.loc[data.index[i], 'volatility_percentile'] = regime_info['vol_percentile']
            
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
            continue
    
    # レジーム列のNaN値を前方に埋める
    regime_cols = ['trend_strength', 'volatility_percentile', 'trend_regime_numeric', 'vol_regime_numeric']
    for col in regime_cols:
        enhanced_data[col] = enhanced_data[col].ffill()
    
    return enhanced_data

# =============================================================================
# 予測関数
# =============================================================================

def predict_with_saved_models(latest_data, reg_model, cls_model, features_reg, features_cls):
    """保存済みモデルで予測実行"""
    try:
        # 回帰用特徴量を準備
        missing_reg_features = [f for f in features_reg if f not in latest_data.columns]
        for feature in missing_reg_features:
            latest_data[feature] = 0.0
        
        X_reg = latest_data[features_reg].values.reshape(1, -1)
        X_reg = np.nan_to_num(X_reg, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 分類用特徴量を準備
        missing_cls_features = [f for f in features_cls if f not in latest_data.columns]
        for feature in missing_cls_features:
            latest_data[feature] = 0.0
        
        X_cls = latest_data[features_cls].values.reshape(1, -1)
        X_cls = np.nan_to_num(X_cls, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 予測実行
        reg_prediction = reg_model.predict(X_reg)[0]
        cls_probability = cls_model.predict_proba(X_cls)[0][1] * 100
        
        # 取引シグナル判定
        if reg_prediction > 0.0 and cls_probability > 50.0:
            signal = "買い"
            signal_emoji = "📈"
            signal_color = "success"
        elif reg_prediction < 0.0 and cls_probability < 50.0:
            signal = "売り"
            signal_emoji = "📉"
            signal_color = "error"
        else:
            signal = "様子見"
            signal_emoji = "⚪"
            signal_color = "warning"
        
        return reg_prediction, cls_probability, signal, signal_emoji, signal_color
        
    except Exception as e:
        st.error(f"❌ 予測エラー: {e}")
        return 0.0, 50.0, "エラー", "❌", "error"

# =============================================================================
# メインアプリケーション
# =============================================================================

def main_app():
    """メインアプリケーション"""
    
    # ヘッダー
    st.markdown('<div class="main-header">📈 株価予測AI（軽量版）</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🤖 LightGBM機械学習モデル
    
    事前訓練済みモデルを使用してリアルタイム株価予測を行います。  
    10年間の歴史データで訓練された高精度モデルです。
    """)
    
    # モデル読み込み
    regression_model, classification_model, features_reg, features_cls = load_saved_models()
    
    if regression_model is None:
        st.error("モデル読み込みに失敗しました。アプリケーションを終了します。")
        return
    
    # サイドバー設定
    st.sidebar.header("⚙️ 設定")
    
    # 市場休み設定
    st.sidebar.subheader("🗓️ 市場休み設定")
    market_choice = st.sidebar.radio(
        "前日の市場状況を選択:",
        (1, 2, 3),
        format_func=lambda x: {
            1: "通常（両市場開場）",
            2: "日本市場のみ休み",
            3: "米国市場のみ休み"
        }[x]
    )
    
    # 市場休み設定の変換
    if market_choice == 2:
        market_holiday = 'japan'
    elif market_choice == 3:
        market_holiday = 'us'
    else:
        market_holiday = None
    
    # 予測実行ボタン
    if st.button("🔮 予測を実行", type="primary", use_container_width=True):
        
        with st.spinner("📊 データを取得しています..."):
            
            # データ取得
            df_stock = download_current_stock_data("1330.T", market_holiday=market_holiday)
            df_market = download_market_indicators()
            
            if df_stock is None or df_market is None:
                st.error("データ取得に失敗しました")
                return
            
            # データ結合
            combined_data = combine_stock_and_market_data(df_stock, df_market)
            if combined_data is None:
                st.error("データ結合に失敗しました")
                return
            
            # 数値変換
            for col in combined_data.columns:
                if col != 'Date':
                    try:
                        combined_data[col] = pd.to_numeric(combined_data[col], errors='coerce')
                    except:
                        pass
            
            # 市場レジーム特徴量の追加
            combined_data = add_regime_features(combined_data)
            
            # テストデータの準備
            drop_cols = ['Date', 'open', 'close', 'volume', 'USD_JPY', 'WTI_Crude',
                        'Gold_future', 'CME_Nikkei_future', 
                        'prevday_WTI_Crude', 'prevday_Gold_future', 'prevday_close_ratio']
            
            test_data = combined_data.drop(columns=[col for col in drop_cols if col in combined_data.columns])
            test_data = test_data.tail(1)
            
        with st.spinner("🤖 AIが予測を計算中..."):
            # 予測実行
            prediction, probability, signal, signal_emoji, signal_color = predict_with_saved_models(
                test_data, regression_model, classification_model, 
                features_reg, features_cls
            )
        
        # 結果表示
        st.markdown("---")
        st.markdown("### 📊 予測結果")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="📈 始値終値比率予測",
                value=f"{prediction:.3f}%",
                delta=f"{'上昇' if prediction > 0 else '下落'}傾向"
            )
        
        with col2:
            st.metric(
                label="📊 上昇確率",
                value=f"{probability:.1f}%",
                delta=f"{'高確率' if probability > 60 else '低確率' if probability < 40 else '中確率'}"
            )
        
        with col3:
            if signal_color == "success":
                st.success(f"{signal_emoji} **{signal}**")
            elif signal_color == "error":
                st.error(f"{signal_emoji} **{signal}**")
            else:
                st.warning(f"{signal_emoji} **{signal}**")
        
        # 詳細情報
        with st.expander("📋 詳細情報"):
            st.markdown(f"""
            **📌 予測の詳細**
            - 回帰モデル予測: {prediction:.4f}%
            - 分類モデル確率: {probability:.2f}%
            - 取引シグナル: {signal}
            
            **⚙️ 設定**
            - 対象銘柄: 1330.T (MAXIS 日経225連動型上場投信)
            - 市場休み設定: {market_choice}
            - 使用特徴量数: 回帰({len(features_reg)})、分類({len(features_cls)})
            """)
    
    # フッター
    st.markdown("---")
    st.markdown("""
    **📝 免責事項:** この予測結果は投資判断の参考情報であり、投資を推奨するものではありません。  
    投資は自己責任で行ってください。
    """)

# =============================================================================
# メイン実行部分
# =============================================================================

def main():
    """アプリのメインエントリーポイント"""
    
    # パスワード認証
    if not check_password():
        return
    
    # メインアプリ実行
    main_app()

if __name__ == "__main__":
    main()