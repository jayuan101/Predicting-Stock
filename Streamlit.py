{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMCjW/hPKm8SqST3B9AC7nx",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/jayuan101/Predicting-Stock/blob/main/Streamlit.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "S1nWjAMeGD_K",
        "outputId": "7d29b14f-7503-48d4-bccd-5bdd1970b599"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: yfinance in /usr/local/lib/python3.10/dist-packages (0.2.41)\n",
            "Collecting streamlit\n",
            "  Downloading streamlit-1.37.1-py2.py3-none-any.whl.metadata (8.5 kB)\n",
            "Requirement already satisfied: pandas>=1.3.0 in /usr/local/lib/python3.10/dist-packages (from yfinance) (2.1.4)\n",
            "Requirement already satisfied: numpy>=1.16.5 in /usr/local/lib/python3.10/dist-packages (from yfinance) (1.26.4)\n",
            "Requirement already satisfied: requests>=2.31 in /usr/local/lib/python3.10/dist-packages (from yfinance) (2.32.3)\n",
            "Requirement already satisfied: multitasking>=0.0.7 in /usr/local/lib/python3.10/dist-packages (from yfinance) (0.0.11)\n",
            "Requirement already satisfied: lxml>=4.9.1 in /usr/local/lib/python3.10/dist-packages (from yfinance) (4.9.4)\n",
            "Requirement already satisfied: platformdirs>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from yfinance) (4.2.2)\n",
            "Requirement already satisfied: pytz>=2022.5 in /usr/local/lib/python3.10/dist-packages (from yfinance) (2024.1)\n",
            "Requirement already satisfied: frozendict>=2.3.4 in /usr/local/lib/python3.10/dist-packages (from yfinance) (2.4.4)\n",
            "Requirement already satisfied: peewee>=3.16.2 in /usr/local/lib/python3.10/dist-packages (from yfinance) (3.17.6)\n",
            "Requirement already satisfied: beautifulsoup4>=4.11.1 in /usr/local/lib/python3.10/dist-packages (from yfinance) (4.12.3)\n",
            "Requirement already satisfied: html5lib>=1.1 in /usr/local/lib/python3.10/dist-packages (from yfinance) (1.1)\n",
            "Requirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.2.2)\n",
            "Requirement already satisfied: blinker<2,>=1.0.0 in /usr/lib/python3/dist-packages (from streamlit) (1.4)\n",
            "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (5.4.0)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.1.7)\n",
            "Requirement already satisfied: packaging<25,>=20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (24.1)\n",
            "Requirement already satisfied: pillow<11,>=7.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (9.4.0)\n",
            "Requirement already satisfied: protobuf<6,>=3.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.20.3)\n",
            "Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (14.0.2)\n",
            "Requirement already satisfied: rich<14,>=10.14.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (13.7.1)\n",
            "Collecting tenacity<9,>=8.1.0 (from streamlit)\n",
            "  Downloading tenacity-8.5.0-py3-none-any.whl.metadata (1.2 kB)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.12.2)\n",
            "Collecting gitpython!=3.1.19,<4,>=3.0.7 (from streamlit)\n",
            "  Downloading GitPython-3.1.43-py3-none-any.whl.metadata (13 kB)\n",
            "Collecting pydeck<1,>=0.8.0b4 (from streamlit)\n",
            "  Downloading pydeck-0.9.1-py2.py3-none-any.whl.metadata (4.1 kB)\n",
            "Requirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.3.3)\n",
            "Collecting watchdog<5,>=2.1.5 (from streamlit)\n",
            "  Downloading watchdog-4.0.2-py3-none-manylinux2014_x86_64.whl.metadata (38 kB)\n",
            "Requirement already satisfied: entrypoints in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.4)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (3.1.4)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (4.23.0)\n",
            "Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.12.1)\n",
            "Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.10/dist-packages (from beautifulsoup4>=4.11.1->yfinance) (2.5)\n",
            "Collecting gitdb<5,>=4.0.1 (from gitpython!=3.1.19,<4,>=3.0.7->streamlit)\n",
            "  Downloading gitdb-4.0.11-py3-none-any.whl.metadata (1.2 kB)\n",
            "Requirement already satisfied: six>=1.9 in /usr/local/lib/python3.10/dist-packages (from html5lib>=1.1->yfinance) (1.16.0)\n",
            "Requirement already satisfied: webencodings in /usr/local/lib/python3.10/dist-packages (from html5lib>=1.1->yfinance) (0.5.1)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.3.0->yfinance) (2.8.2)\n",
            "Requirement already satisfied: tzdata>=2022.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.3.0->yfinance) (2024.1)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.31->yfinance) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.31->yfinance) (3.7)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.31->yfinance) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.31->yfinance) (2024.7.4)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (2.16.1)\n",
            "Collecting smmap<6,>=3.0.1 (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit)\n",
            "  Downloading smmap-5.0.1-py3-none-any.whl.metadata (4.3 kB)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.5)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (24.2.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.12.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.35.1)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.20.0)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)\n",
            "Downloading streamlit-1.37.1-py2.py3-none-any.whl (8.7 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m8.7/8.7 MB\u001b[0m \u001b[31m27.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading GitPython-3.1.43-py3-none-any.whl (207 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m207.3/207.3 kB\u001b[0m \u001b[31m15.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading pydeck-0.9.1-py2.py3-none-any.whl (6.9 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m6.9/6.9 MB\u001b[0m \u001b[31m70.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading tenacity-8.5.0-py3-none-any.whl (28 kB)\n",
            "Downloading watchdog-4.0.2-py3-none-manylinux2014_x86_64.whl (82 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m82.9/82.9 kB\u001b[0m \u001b[31m6.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading gitdb-4.0.11-py3-none-any.whl (62 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m62.7/62.7 kB\u001b[0m \u001b[31m4.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading smmap-5.0.1-py3-none-any.whl (24 kB)\n",
            "Installing collected packages: watchdog, tenacity, smmap, pydeck, gitdb, gitpython, streamlit\n",
            "  Attempting uninstall: tenacity\n",
            "    Found existing installation: tenacity 9.0.0\n",
            "    Uninstalling tenacity-9.0.0:\n",
            "      Successfully uninstalled tenacity-9.0.0\n",
            "Successfully installed gitdb-4.0.11 gitpython-3.1.43 pydeck-0.9.1 smmap-5.0.1 streamlit-1.37.1 tenacity-8.5.0 watchdog-4.0.2\n"
          ]
        }
      ],
      "source": [
        "!pip install yfinance streamlit\n",
        "import streamlit as st\n",
        "import yfinance as yf\n",
        "import tensorflow as tf\n",
        "import numpy as np\n",
        "from tensorflow.keras.layers import LSTM, Dense, Dropout\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import r2_score\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "st.title('Stock Price Prediction')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zDml5G_tGGCV",
        "outputId": "bbbefa77-9de6-4c29-99f9-feb8feb7ff04"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2024-08-18 01:47:18.623 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py [ARGUMENTS]\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "DeltaGenerator()"
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "stock_symbol = st.text_input('Enter Stock Symbol', 'AAPL')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OhIjVnLTGGFE",
        "outputId": "a21c6468-f518-43a6-8b18-7edf1164ca53"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2024-08-18 01:47:20.577 Session state does not function when running a script without `streamlit run`\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "stock_symbol = st.text_input('Enter Stock Symbol', 'GOOG')"
      ],
      "metadata": {
        "id": "DqYqDBq4GZH7"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "stock_symbol = st.text_input('Enter Stock Symbol', 'MSFT')"
      ],
      "metadata": {
        "id": "mqdAnH_9GZKV"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "stock_symbol = st.text_input('Enter Stock Symbol', 'AMAN')"
      ],
      "metadata": {
        "id": "tlJ8a8S2GZNN"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "stock_symbol = st.text_input('Enter Stock Symbol', 'SATS')"
      ],
      "metadata": {
        "id": "b8wqjMPcGZPl"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "if stock_symbol:\n",
        "    stock_data = yf.download(stock_symbol, start='2020-01-01', end='2023-01-01')\n",
        "    if not stock_data.empty:\n",
        "        st.subheader('Stock Data')\n",
        "        st.write(stock_data[['Open', 'High', 'Low', 'Close']]) # Pass a list of column names\n",
        "    else:\n",
        "        st.warning('No stock data available for the entered symbol.')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hNoW8OtdHHa8",
        "outputId": "fec349d4-d6d2-43cd-a75e-e5da20550fe7"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r[*********************100%%**********************]  1 of 1 completed\n"
          ]
        }
      ]
    }
  ]
}