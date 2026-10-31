"""文本处理、清洗、去重、实体链接的单元测试。"""

from servicekb.models import Entity, EntityRelation
from servicekb.processing.cleaning import clean_text, redact_pii, strip_email_noise
from servicekb.processing.dedup import cluster_near_duplicates, hamming, simhash
from servicekb.processing.entities import EntityLinker
from servicekb.processing.text import (
    estimate_tokens,
    normalize_key,
    normalize_text,
    split_sentences,
    tokenize,
)


def test_normalize_fullwidth_but_keep_chinese_punctuation():
    assert normalize_text("ＡＳ－１１０Ｋ　报Ｅ０３７，怎么办？") == "AS-110K 报E037，怎么办？"
    assert normalize_text("a​b\r\n\r\n\r\n\r\nc") == "ab\n\nc"


def test_normalize_key_unifies_spellings():
    assert normalize_key("as-110k hv") == normalize_key("AS110KHV") == normalize_key("AS_110K-HV") == "AS110KHV"


def test_split_sentences_does_not_break_versions_or_part_numbers():
    s = split_sentences("固件升级到 V2.3.0 以上。更换 310-04521-B. Then restart; done")
    assert s == ["固件升级到 V2.3.0 以上。", "更换 310-04521-B.", "Then restart;", "done"]


def test_tokenize_keeps_identifiers_whole():
    toks = tokenize("更换风扇 310-04521-B 后 E051 消失，固件 V2.3.1")
    assert "310-04521-b" in toks and "e051" in toks and "v2.3.1" in toks
    assert "风扇" in toks


def test_estimate_tokens_mixed():
    assert estimate_tokens("绝缘阻抗") == 4
    assert estimate_tokens("hello world") == 3


def test_strip_email_noise():
    raw = (
        "请问E037怎么处理？\n\nBest regards,\n王工\n"
        "-----Original Message-----\nFrom: someone\n旧邮件内容"
    )
    assert strip_email_noise(raw) == "请问E037怎么处理？"
    raw2 = "已更换风扇，问题解决。\n> 之前的引用\nThis email and any attachments are confidential blah"
    assert strip_email_noise(raw2) == "已更换风扇，问题解决。"


def test_redact_pii_keeps_part_numbers():
    t = redact_pii("联系 13800138000 或 a.b@example.com，料号 310-04521-B，序列号 AS2301234567")
    assert "[PHONE]" in t and "[EMAIL]" in t
    assert "310-04521-B" in t and "AS2301234567" in t
    assert clean_text("电话：+86 13912345678") == "电话：[PHONE]"


def test_simhash_near_duplicates():
    a = simhash("逆变器每天早上报E037绝缘阻抗低，中午自动恢复，最近一直下雨")
    b = simhash("逆变器每天早上报E037绝缘阻抗低，中午自动恢复，最近一直在下雨")
    c = simhash("通讯板更换后WiFi无法连接路由器")
    assert hamming(a, b) < hamming(a, c)
    assert cluster_near_duplicates([("x", a), ("y", a), ("z", c)], 3) == [["x", "y"]]


def _linker() -> EntityLinker:
    ents = [
        Entity("model:AS-110K-HV", "product_model", "AS-110K-HV", aliases=["AS110K", "110K-HV"]),
        Entity("part:310-04521-A", "part", "310-04521-A", "散热风扇组件", ["风扇总成"], {"status": "停产"}),
        Entity("part:310-04521-B", "part", "310-04521-B", "散热风扇组件(带防尘网)", [], {"status": "在售"}),
        Entity("fault:E051", "fault_code", "E051", "风扇故障"),
    ]
    rels = [EntityRelation("part:310-04521-A", "substituted_by", "part:310-04521-B")]
    return EntityLinker(ents, rels)


def test_entity_linking_aliases_and_unknown_parts():
    lk = _linker()
    ms = {m.entity_id: m for m in lk.extract("as-110k 报 e051，换了风扇总成，另有料号 999-99999-Z")}
    assert "model:AS-110K-HV" in ms
    assert "fault:E051" in ms
    assert "part:310-04521-A" in ms  # 通过中文别名"风扇总成"
    assert ms["part:999-99999-Z"].known is False  # 物料主数据里没有 → 进数据质量报告


def test_entity_longest_match_with_fullwidth_parens():
    lk = _linker()
    assert lk.link("请更换散热风扇组件（带防尘网）") == ["part:310-04521-B"]


def test_expand_substitutes_and_facts():
    lk = _linker()
    assert lk.expand(["part:310-04521-A"]) == ["part:310-04521-A", "part:310-04521-B"]
    facts = lk.facts_for(["part:310-04521-A"])
    assert "已被 310-04521-B 替代" in facts[0] and "停产" in facts[0]
    assert lk.resolve_model("110k-hv") == "AS-110K-HV"
