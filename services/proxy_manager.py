import os
import time
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

def get_config(name: str, default=None):
    """
    Read config from:
    1. environment variables / .env
    2. Streamlit root-level secrets
    3. Streamlit [general] secrets with lowercase key
    """
    value = os.getenv(name)
    if value is not None:
        return value

    try:
        import streamlit as st

        # root-level secrets:
        # WEBSHARE_API_TOKEN = "xxx"
        if name in st.secrets:
            return st.secrets[name]

        # [general] secrets:
        # webshare_api_token = "xxx"
        general_key = name.lower()
        if "general" in st.secrets and general_key in st.secrets["general"]:
            return st.secrets["general"][general_key]

    except Exception:
        pass

    return default
class ProxyManager:
    def __init__(self):
        self.api_token = get_config("WEBSHARE_API_TOKEN")
        self.mode = get_config("WEBSHARE_PROXY_MODE", "direct")

        self.exclude_countries = self._parse_country_list(
            get_config("PROXY_EXCLUDE_COUNTRIES", "US")
        )

        self.preferred_countries = self._parse_country_list(
            get_config("PROXY_PREFERRED_COUNTRIES", "JP,GB")
        )

        self.health_check_url = get_config(
            "PROXY_HEALTH_CHECK_URL",
            "https://api.binance.com/api/v3/time",
        )

        self.timeout = int(get_config("PROXY_TIMEOUT_SECONDS", "8"))
        self.refresh_interval = int(get_config("PROXY_REFRESH_MINUTES", "30")) * 60

        self.last_refresh_time = 0.0
        self.proxy_items: List[Dict] = []
        self.active_proxy: Optional[Dict[str, str]] = None
        self.active_proxy_meta: Optional[Dict] = None

    
    def _parse_country_list(self, value: str) -> set[str]:
        return {
            item.strip().upper()
            for item in value.split(",")
            if item.strip()
        }

    def fetch_webshare_proxies(self) -> List[Dict]:
        if not self.api_token:
            raise RuntimeError("WEBSHARE_API_TOKEN is not configured.")

        url = "https://proxy.webshare.io/api/v2/proxy/list/"

        headers = {
            "Authorization": f"Token {self.api_token}"
        }

        params = {
            "mode": self.mode,
            "page_size": 100,
        }

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        return data.get("results", [])

    def filter_proxies(self, proxy_items: List[Dict]) -> List[Dict]:
        filtered = []

        for item in proxy_items:
            country_code = str(item.get("country_code", "")).upper()
            city_name = item.get("city_name")
            is_valid = item.get("valid") is True

            if not is_valid:
                continue

            if country_code in self.exclude_countries:
                continue

            if self.preferred_countries and country_code not in self.preferred_countries:
                continue

            if not city_name:
                continue

            filtered.append(item)

        return filtered

    def build_requests_proxy(self, proxy_item: Dict) -> Dict[str, str]:
        host = proxy_item.get("proxy_address")
        port = proxy_item.get("port")
        username = proxy_item.get("username")
        password = proxy_item.get("password")

        if not host or not port:
            raise ValueError("Invalid proxy item: missing proxy_address or port.")

        if username and password:
            proxy_url = f"http://{username}:{password}@{host}:{port}"
        else:
            proxy_url = f"http://{host}:{port}"

        return {
            "http": proxy_url,
            "https": proxy_url,
        }

    def is_proxy_working(self, requests_proxy: Dict[str, str]) -> bool:
        try:
            response = requests.get(
                self.health_check_url,
                proxies=requests_proxy,
                timeout=self.timeout,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def refresh_proxy(self, force: bool = False) -> Optional[Dict[str, str]]:
        now = time.time()

        if (
            not force
            and self.active_proxy is not None
            and now - self.last_refresh_time < self.refresh_interval
        ):
            return self.active_proxy

        raw_items = self.fetch_webshare_proxies()
        candidate_items = self.filter_proxies(raw_items)

        self.proxy_items = candidate_items
        self.last_refresh_time = now

        for item in candidate_items:
            try:
                requests_proxy = self.build_requests_proxy(item)

                if self.is_proxy_working(requests_proxy):
                    self.active_proxy = requests_proxy
                    self.active_proxy_meta = {
                        "country_code": item.get("country_code"),
                        "city_name": item.get("city_name"),
                        "proxy_address": item.get("proxy_address"),
                        "port": item.get("port"),
                        "last_verification": item.get("last_verification"),
                    }
                    return self.active_proxy

            except Exception:
                continue

        self.active_proxy = None
        self.active_proxy_meta = None
        return None

    def get_active_proxy(self) -> Optional[Dict[str, str]]:
        return self.refresh_proxy(force=False)

    def rotate_proxy(self) -> Optional[Dict[str, str]]:
        return self.refresh_proxy(force=True)

    def get_status(self) -> Dict:
        return {
            "has_api_token": bool(self.api_token),
            "active": self.active_proxy is not None,
            "meta": self.active_proxy_meta,
            "candidate_count": len(self.proxy_items),
            "exclude_countries": sorted(self.exclude_countries),
            "preferred_countries": sorted(self.preferred_countries),
        }


proxy_manager = ProxyManager()