"""
Mehrsprachige Keyword-Listen für Sentiment-Analyse.
Score: +1 = bullish, -1 = bearish
Für nicht-englische Headlines wo FinBERT nicht greift.
"""

KEYWORDS = {
    # Deutsch
    "de": {
        "bullish": [
            "anstieg", "angestiegen", "steigt", "rallye", "rally", "hoch", "höchststand",
            "rekord", "durchbruch", "bullish", "gewinne", "gewinn", "erholung", "erholt",
            "zugelassen", "genehmigt", "partnerschaft", "adoption", "upgrade", "erfolg",
            "neues hoch", "ausbruch", "nachfrage", "wachstum",
        ],
        "bearish": [
            "absturz", "abgestürzt", "einbruch", "crash", "fällt", "gefallen", "tief",
            "tiefstand", "bearish", "verlust", "verluste", "panik", "angst", "hack",
            "gehackt", "betrug", "verbot", "regulierung", "klage", "insolvenz", "bankrott",
            "manipulation", "warnung", "rückgang", "ausverkauf",
        ],
    },
    # Chinesisch (Vereinfacht)
    "zh": {
        "bullish": [
            "突破", "上涨", "暴涨", "飙升", "新高", "历史新高", "看涨", "利好",
            "反弹", "牛市", "大涨", "强势", "创新高", "获批", "合作", "升级",
            "增长", "需求", "买入",
        ],
        "bearish": [
            "暴跌", "下跌", "崩盘", "大跌", "新低", "看跌", "利空", "恐慌",
            "熊市", "抛售", "清算", "爆仓", "黑客", "攻击", "被盗", "诈骗",
            "禁止", "监管", "起诉", "破产", "跑路", "崩溃", "警告",
        ],
    },
    # Japanisch
    "ja": {
        "bullish": [
            "急騰", "上昇", "高騰", "突破", "最高値", "過去最高", "強気", "好材料",
            "反発", "回復", "承認", "提携", "採用", "成功", "上場", "需要",
            "成長", "買い",
        ],
        "bearish": [
            "急落", "下落", "暴落", "大暴落", "最安値", "弱気", "悪材料", "恐怖",
            "警戒", "売り", "清算", "ハッキング", "盗難", "詐欺", "規制",
            "禁止", "訴訟", "破綻", "破産", "崩壊", "暴落", "警告",
        ],
    },
    # Koreanisch
    "ko": {
        "bullish": [
            "급등", "상승", "폭등", "돌파", "최고", "신고가", "강세", "호재",
            "반등", "회복", "승인", "파트너십", "채택", "성공", "상장", "수요",
            "성장", "매수",
        ],
        "bearish": [
            "급락", "하락", "폭락", "붕괴", "최저", "약세", "악재", "공포",
            "패닉", "매도", "청산", "해킹", "도난", "사기", "규제",
            "금지", "소송", "파산", "경고", "대폭락",
        ],
    },
    # Russisch
    "ru": {
        "bullish": [
            "рост", "взлет", "взлетел", "прорыв", "рекорд", "максимум", "бычий",
            "ралли", "отскок", "восстановление", "одобрен", "партнерство", "принят",
            "спрос", "рост",
        ],
        "bearish": [
            "обвал", "падение", "рухнул", "крах", "минимум", "медвежий", "паника",
            "распродажа", "ликвидация", "взлом", "украден", "мошенничество",
            "запрет", "регулирование", "иск", "банкротство", "крах",
        ],
    },
    # Türkisch
    "tr": {
        "bullish": [
            "yükseliş", "artış", "yükseldi", "rekor", "zirve", "boğa", "ralli",
            "toparlanma", "onay", "ortaklık", "benimseme", "başarı", "talep",
            "büyüme",
        ],
        "bearish": [
            "düşüş", "çöküş", "çöktü", "sert düşüş", "dip", "ayı", "panik",
            "satış", "tasfiye", "hack", "çalındı", "dolandırıcılık", "yasak",
            "düzenleme", "dava", "iflas", "uyarı",
        ],
    },
    # Spanisch
    "es": {
        "bullish": [
            "sube", "subida", "alza", "récord", "máximo", "alcista", "rally",
            "recuperación", "aprobado", "asociación", "adopción", "éxito", "demanda",
            "crecimiento",
        ],
        "bearish": [
            "cae", "caída", "desploma", "colapso", "mínimo", "bajista", "pánico",
            "venta masiva", "liquidación", "hackeo", "robado", "estafa", "prohibición",
            "regulación", "demanda judicial", "quiebra", "advertencia",
        ],
    },
}


def keyword_score(text: str, lang: str) -> float | None:
    """Bewertet Text anhand von Keywords. Returns -1.0 bis +1.0 oder None wenn keine Keywords gefunden."""
    if lang not in KEYWORDS:
        return None

    text_lower = text.lower()
    bull_count = sum(1 for kw in KEYWORDS[lang]["bullish"] if kw in text_lower)
    bear_count = sum(1 for kw in KEYWORDS[lang]["bearish"] if kw in text_lower)

    total = bull_count + bear_count
    if total == 0:
        return None

    # Score: -1.0 (all bearish) to +1.0 (all bullish)
    return (bull_count - bear_count) / total
