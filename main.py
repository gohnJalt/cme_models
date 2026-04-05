from market_dashboard import MarketDashboard
from geo_dashboard import GeoTradeDashboard
import yfinance as yf
from markov_regime import RegimeModel


if __name__ == "__main__":
    dashboard_market = MarketDashboard()
    dashboard_market.refresh()
    print(dashboard_market.report())
    dashboard_market.save_snapshot()
    dashboard_geo = GeoTradeDashboard()
    dashboard_geo.refresh()
    print(dashboard_geo.report())
    dashboard_geo.save_snapshot()

    print("Fetching VIX, Brent, Gold...")
    raw = yf.download(['^VIX', 'BZ=F', 'GC=F'], period='3y',
                      interval='1d', progress=False, auto_adjust=True)
    vix = raw['Close']['^VIX'].dropna()
    brent = raw['Close']['BZ=F'].dropna()
    gold = raw['Close']['GC=F'].dropna()

    model = RegimeModel().fit(
        driver_series=vix,
        target_series={'BZ=F': brent, 'GC=F': gold},
    )
    print(model.report())