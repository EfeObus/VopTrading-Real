# FreqUI

Voptrade provides a builtin webserver, which can serve [FreqUI](https://github.com/freqtrade/frequi), the voptrade frontend.

By default, the UI is automatically installed as part of the installation (script, docker).
freqUI can also be manually installed by using the `voptrade install-ui` command.
This same command can also be used to update freqUI to new releases.

Once the bot is started in trade / dry-run mode (with `voptrade trade`) - the UI will be available under the configured API port (by default `http://127.0.0.1:8080`).

??? Note "Looking to contribute to freqUI?"
    Developers should not use this method, but instead clone the corresponding use the method described in the [freqUI repository](https://github.com/freqtrade/frequi) to get the source-code of freqUI. A working installation of node will be required to build the frontend.

!!! tip "freqUI is not required to run voptrade"
    freqUI is an optional component of voptrade, and is not required to run the bot.
    It is a frontend that can be used to monitor the bot and to interact with it - but voptrade itself will work perfectly fine without it.

## Configuration

FreqUI does not have it's own configuration file - but assumes a working setup for the [rest-api](rest-api.md) is available.
Please refer to the corresponding documentation page to get setup with freqUI

## UI

FreqUI is a modern, responsive web application that can be used to monitor and interact with your bot.

FreqUI provides a light, as well as a dark theme.
Themes can be easily switched via a prominent button at the top of the page.

### Login

!!! Hint "CORS"
    If the UI is running on a different port than the API, you may see a CORS error on login - see [CORS](#cors) for how to configure this correctly.

### Trade view

The trade view allows you to visualize the trades that the bot is making and to interact with the bot.
On this page, you can also interact with the bot by starting and stopping it and - if configured - force trade entries and exits.

### Dashboard

The dashboard view provides an overview of the bot's performance and status.
If multiple bots are connected, the dashboard will show an overview of all connected bots, allowing you to easily switch between them or show just a subset of available bots.

#### Wallet Balance

New in voptrade 2026.4: This shows the balance of the bot over time.

Compared to the "cumulative Profit" chart, this chart will show the actual balance of the bot over time, including unrealized profit and losses, as well as deposits and withdrawals.

Historic data has re-populated based on available exchange data - however is assumed to be best-effort and may not be 100% accurate.
More specifically, it won't cover deposits and withdrawals, and will assume a starting balance of current balance - profit/losses.

For clarity - a "Capture start" marker line is shown on the chart, which indicates the point at which the migration to the new wallet balance tracking system happened.
Only beyond this point, the wallet balance is expected to be accurate.

### Plot Configurator

FreqUI Plots can be configured either via a `plot_config` configuration object in the strategy (which can be loaded via "from strategy" button) or via the UI.
Multiple plot configurations can be created and switched at will - allowing for flexible, different views into your charts.

The plot configuration can be accessed via the "Plot Configurator" (Cog icon) button in the top right corner of the trade view.

### Settings

Several UI related settings can be changed by accessing the settings page.

Things you can change (among others):

* Timezone of the UI
* Visualization of open trades as part of the favicon (browser tab)
* Candle colors (up/down -> red/green)
* Enable / disable in-app notification types

## Webserver mode

when voptrade is started in [webserver mode](utils.md#webserver-mode) (voptrade started with `voptrade webserver`), the webserver will start in a special mode allowing for additional features, for example:

* Downloading data
* Testing pairlists
* [Backtesting strategies](#backtesting)
* ... to be expanded

### Backtesting

When voptrade is started in [webserver mode](utils.md#webserver-mode) (voptrade started with `voptrade webserver`), the backtesting view becomes available.
This view allows you to backtest strategies and visualize the results.

You can also load and visualize previous backtest results, as well as compare the results with each other.


--8<-- "includes/cors.md"
