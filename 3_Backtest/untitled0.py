




actions = bt_results.copy()


actions['DAY_TRADING_HIGH'] = \
    actions[['OPEN', 'HIGH', 'LOW', 'CLOSE']].max(axis=1)

actions['DAY_TRADING_LOW'] = \
    actions[['OPEN', 'HIGH', 'LOW', 'CLOSE']].min(axis=1)
    
actions['DAY_TRADING_SIGNAL'] = \
    actions['DAY_TRADING_HIGH'] - actions['DAY_TRADING_LOW']