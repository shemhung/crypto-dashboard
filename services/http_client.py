import requests
from streamlit import status

from services.proxy_manager import proxy_manager


def get_with_auto_proxy(url, params=None, headers=None, timeout=10):
    """
    Send GET request with active non-US Webshare proxy.
    If the request fails, rotate proxy and retry once.
    """

    proxy = None

    try:
        proxy = proxy_manager.get_active_proxy()
        status = proxy_manager.get_status()
        print("[ProxyManager] Active proxy meta:", status.get("meta"))
    except Exception as e:
        print(f"[ProxyManager] No active proxy, fallback to direct request: {e}")

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            proxies=proxy,
            timeout=timeout,
        )
        response.raise_for_status()
        return response

    except requests.RequestException as first_error:
        print(f"[HTTP] First request failed: {first_error}")
        print("[HTTP] Rotating proxy...")

        try:
            new_proxy = proxy_manager.rotate_proxy()
            status = proxy_manager.get_status()
            print("[ProxyManager] New proxy meta:", status.get("meta"))

        except Exception as e:
            print(f"[ProxyManager] Proxy rotation failed: {e}")
            raise first_error

        # 4. 換 proxy 後再試一次
        response = requests.get(
            url,
            params=params,
            headers=headers,
            proxies=new_proxy,
            timeout=timeout,
        )
        response.raise_for_status()
        return response