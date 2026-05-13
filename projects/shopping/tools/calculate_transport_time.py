"""calculate_transport_time — region-to-region delivery estimate with
provider modifiers. Ports the reference's hardcoded matrices verbatim."""
from __future__ import annotations

import json

from platform_core.tools import register_tool
from . import _db

NAME = "calculate_transport_time"

SCHEMA = {
    "name": NAME,
    "description": (
        "Estimates delivery days for a product from its shipping origin to a "
        "destination province. Returns an integer day count, never less than 1."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "product_id": {
                "type": "string",
                "description": "Product to ship.",
            },
            "destination_address": {
                "type": "string",
                "description": (
                    "Destination province name in pinyin (e.g. 'guangdong', "
                    "'beijing', 'shanghai')."
                ),
            },
            "provider": {
                "type": "string",
                "description": (
                    "Optional. Override the product's default shipping provider. "
                    "Supported: 'sf_express', 'jd_logistics', 'yto_express', "
                    "'zto_express', 'sto_express', 'yunda_express', 'cainiao', "
                    "'china_post', 'ems', 'deppon_express', 'default'."
                ),
            },
        },
        "required": ["product_id", "destination_address"],
    },
}


_PROVINCE_ALIASES = {
    "beijing": ["beijing", "bj", "北京"],
    "shanghai": ["shanghai", "sh", "上海"],
    "tianjin": ["tianjin", "tj", "天津"],
    "chongqing": ["chongqing", "cq", "重庆"],
    "hebei": ["hebei", "ji", "河北"],
    "shanxi": ["shanxi", "jin", "山西"],
    "liaoning": ["liaoning", "liao", "辽宁"],
    "jilin": ["jilin", "ji_ln", "吉林"],
    "heilongjiang": ["heilongjiang", "hei", "黑龙江"],
    "jiangsu": ["jiangsu", "su", "江苏"],
    "zhejiang": ["zhejiang", "zhe", "浙江"],
    "anhui": ["anhui", "wan", "安徽"],
    "fujian": ["fujian", "min", "福建"],
    "jiangxi": ["jiangxi", "gan", "江西"],
    "shandong": ["shandong", "lu", "山东"],
    "henan": ["henan", "yu", "河南"],
    "hubei": ["hubei", "e", "湖北"],
    "hunan": ["hunan", "xiang", "湖南"],
    "guangdong": ["guangdong", "yue", "gd", "广东"],
    "hainan": ["hainan", "qiong", "海南"],
    "sichuan": ["sichuan", "chuan", "shu", "四川"],
    "guizhou": ["guizhou", "qian", "gui_gz", "贵州"],
    "yunnan": ["yunnan", "yun", "dian", "云南"],
    "shaanxi": ["shaanxi", "shan", "qin", "陕西"],
    "gansu": ["gansu", "gan_gs", "甘肃"],
    "qinghai": ["qinghai", "qing", "青海"],
    "inner mongolia": ["inner mongolia", "neimenggu", "meng", "内蒙古"],
    "guangxi": ["guangxi", "gui", "广西"],
    "tibet": ["tibet", "xizang", "zang", "西藏"],
    "ningxia": ["ningxia", "ning", "宁夏"],
    "xinjiang": ["xinjiang", "xin", "新疆"],
    "hongkong": ["hongkong", "hk", "xianggang", "香港"],
    "macau": ["macau", "mo", "aomen", "澳门"],
    "taiwan": ["taiwan", "tw", "台湾"],
}

_PROVINCE_NORMALIZATION_MAP = {
    alias: std for std, aliases in _PROVINCE_ALIASES.items() for alias in aliases
}

_REGION_MAP = {
    "beijing": "NC", "tianjin": "NC", "hebei": "NC", "shanxi": "NC", "inner mongolia": "NC",
    "liaoning": "NE", "jilin": "NE", "heilongjiang": "NE",
    "shanghai": "EC", "jiangsu": "EC", "zhejiang": "EC", "anhui": "EC",
    "fujian": "EC", "jiangxi": "EC", "shandong": "EC",
    "henan": "CC", "hubei": "CC", "hunan": "CC",
    "guangdong": "SC", "guangxi": "SC", "hainan": "SC",
    "hongkong": "SC", "macau": "SC", "taiwan": "SC",
    "sichuan": "SW", "chongqing": "SW", "guizhou": "SW",
    "yunnan": "SW", "tibet": "SW",
    "shaanxi": "NW", "gansu": "NW", "qinghai": "NW", "ningxia": "NW", "xinjiang": "NW",
}

_BASE_REGION_TIME = {
    "NC": {"NC": 1, "NE": 2, "EC": 2, "CC": 2, "SC": 3, "SW": 3, "NW": 3},
    "NE": {"NC": 2, "NE": 1, "EC": 3, "CC": 3, "SC": 4, "SW": 4, "NW": 4},
    "EC": {"NC": 2, "NE": 3, "EC": 1, "CC": 2, "SC": 2, "SW": 3, "NW": 4},
    "CC": {"NC": 2, "NE": 3, "EC": 2, "CC": 1, "SC": 2, "SW": 2, "NW": 3},
    "SC": {"NC": 3, "NE": 4, "EC": 2, "CC": 2, "SC": 1, "SW": 3, "NW": 4},
    "SW": {"NC": 3, "NE": 4, "EC": 3, "CC": 2, "SC": 3, "SW": 1, "NW": 3},
    "NW": {"NC": 3, "NE": 4, "EC": 4, "CC": 3, "SC": 4, "SW": 3, "NW": 1},
}

_PROVIDER_MODIFIERS = {
    "sf express": -2, "sf": -2, "sf_express": -2,
    "jd logistics": -1, "jd": -1, "jd_logistics": -1,
    "yto express": 1, "yto": 0, "yto_express": 1,
    "zto express": 1, "zto": 0, "zto_express": 1,
    "sto express": 1, "sto": 0, "sto_express": 1,
    "yunda express": 1, "yunda": 0, "yunda_express": 1,
    "cainiao": 1,
    "china post": 2, "china_post": 2,
    "ems": 0,
    "deppon express": 0, "deppon": 0, "deppon_express": 0,
    "default": 0,
}

_REMOTE_PROVINCES = {"tibet", "xinjiang", "qinghai", "inner mongolia"}


def _normalize(address: str):
    if not address:
        return None
    s = address.lower().replace(" ", "").replace("province", "").replace("city", "")
    if s in _PROVINCE_NORMALIZATION_MAP:
        return _PROVINCE_NORMALIZATION_MAP[s]
    for alias, std in _PROVINCE_NORMALIZATION_MAP.items():
        if alias in s:
            return std
    return None


def run(product_id=None, destination_address=None, provider=None) -> str:
    products = _db.products_by_id()
    if not products:
        return _db.db_missing_sentinel(NAME)
    if not product_id:
        return json.dumps({"error": "product_id is required"}, ensure_ascii=False)
    if not destination_address:
        return json.dumps(
            {"error": "destination_address is required"}, ensure_ascii=False
        )
    product = products.get(product_id)
    if not product:
        return json.dumps(
            {"error": f"Product with ID {product_id!r} not found."},
            ensure_ascii=False,
        )
    shipping = product.get("shipping_info", {}) or {}
    origin = shipping.get("origin")
    if not origin:
        return json.dumps(
            {"error": f"Shipping origin not found for product {product_id!r}."},
            ensure_ascii=False,
        )
    chosen_provider = (provider or shipping.get("provider") or "default").lower()
    origin_p = _normalize(origin)
    dest_p = _normalize(destination_address)
    if not origin_p:
        return json.dumps(
            {"error": f"Could not determine a valid province from origin address: {origin!r}."},
            ensure_ascii=False,
        )
    if not dest_p:
        return json.dumps(
            {
                "error": (
                    f"Could not determine a valid province from destination "
                    f"address: {destination_address!r}. Use a Chinese province name in pinyin."
                )
            },
            ensure_ascii=False,
        )
    origin_r = _REGION_MAP.get(origin_p)
    dest_r = _REGION_MAP.get(dest_p)
    if not origin_r or not dest_r:
        return json.dumps(
            {"error": "Could not map provinces to geographical regions."},
            ensure_ascii=False,
        )
    base = _BASE_REGION_TIME[origin_r][dest_r]
    if origin_p in _REMOTE_PROVINCES or dest_p in _REMOTE_PROVINCES:
        base += 2
    modifier = _PROVIDER_MODIFIERS.get(chosen_provider, _PROVIDER_MODIFIERS["default"])
    days = max(1, base + modifier)
    return json.dumps(
        {
            "product_id": product_id,
            "origin": origin,
            "destination": destination_address,
            "estimated_delivery_days": days,
        },
        ensure_ascii=False,
    )


register_tool(NAME, SCHEMA, run)
