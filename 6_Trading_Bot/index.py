


from configparser import ConfigParser
from fugle_trade.sdk import SDK
# from fugle_trade.sdk import reset_password
from fugle_trade.order import OrderObject
from fugle_trade.constant import (APCode, Trade, PriceFlag, BSFlag, Action)

config = ConfigParser()
# config.read('./config.simulation.ini')
config.read(r'/Users/aron/Documents/GitHub/Stock_Forecast/6_Trading_Bot/config.simulation.ini')
sdk = SDK(config)
sdk.login()
# sdk.reset_password()


order = OrderObject(
    buy_sell = Action.Buy,
    price_flag = PriceFlag.LimitDown,
    price = None,
    stock_no = "2884",
    quantity = 1,
)
sdk.place_order(order)
print("Your order has been placed successfully.")