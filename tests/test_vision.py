"""拍照识别备件：图片处理、OCR、图像向量、识别融合、增量入库、问答、API、评估。"""

import io
import json
import shutil
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from servicekb.vision.embedders import HandcraftedImageEmbedder, HTTPImageEmbedder
from servicekb.vision.images import has_content, load_image, make_crops, pad_to_square
from servicekb.vision.ocr import MetadataOCR, normalize_ocr_text

ROOT = Path(__file__).resolve().parents[1]
PHOTOS = ROOT / "data" / "sample" / "photos"
REFS = ROOT / "data" / "sample" / "part_images"


# --------------------------------------------------------------------------- 基础处理


def test_load_image_applies_exif_orientation():
    buf = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6  # 手机竖拍：像素是横的，EXIF 说"顺时针转 90°"
    Image.new("RGB", (40, 20), (200, 10, 10)).save(buf, format="JPEG", exif=exif)
    assert load_image(buf.getvalue()).size == (20, 40)


def test_pad_to_square_and_tiling_skips_blank_tiles():
    assert pad_to_square(Image.new("RGB", (300, 100), "white")).size == (300, 300)
    big = Image.new("RGB", (3600, 3600), (128, 128, 128))
    assert len(make_crops(big)) == 1  # 纯色切块不参与匹配
    big.paste(Image.open(PHOTOS / "site_310-04522-A_nolabel.jpg"), (2800, 3000))
    crops = make_crops(big, grid=2, min_side=1200)
    assert len(crops) == 2 and has_content(crops[1])  # 整图 + 有零件的那一块


def test_normalize_ocr_text_splits_glued_keys_and_fixes_digits():
    assert normalize_ocr_text("P/N310-04521-B\nSNAS112203004561") == "P/N 310-04521-B\nSN AS112203004561"
    assert normalize_ocr_text("PN:3IO-O4521-A") == "PN 310-04521-A"
    assert normalize_ocr_text("料号：410-20031-D") == "料号 410-20031-D"


def test_metadata_ocr_reads_sample_label():
    img = load_image(PHOTOS / "site_310-04521-B_label.jpg")
    assert "310-04521-B" in MetadataOCR().read(img)
    assert MetadataOCR().read(load_image(PHOTOS / "site_310-04521-B_nolabel.jpg")) == ""


def test_rapidocr_reads_real_pixels():
    pytest.importorskip("rapidocr_onnxruntime")
    from servicekb.vision.ocr import RapidOCREngine

    text, lines = RapidOCREngine().read_with_boxes(load_image(PHOTOS / "site_220-11009-A_label.jpg"))
    assert "220-11009-A" in normalize_ocr_text(text)
    assert lines and all(len(box) == 4 for _, box in lines)


# --------------------------------------------------------------------------- 图像向量


def test_handcrafted_embedder_matches_rotated_reference():
    e = HandcraftedImageEmbedder()
    refs = sorted(REFS.glob("*/front.png"))
    ref_vecs = np.asarray(e.embed_images([load_image(p) for p in refs]))
    query = load_image(REFS / "310-04522-A" / "render_rotated.png")
    sims = ref_vecs @ np.asarray(e.embed_images([query])[0])
    assert refs[int(np.argmax(sims))].parent.name == "310-04522-A"
    assert abs(float(np.linalg.norm(ref_vecs[0])) - 1.0) < 1e-5


def test_siglip_embedder_with_tiny_random_model():
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from servicekb.vision.embedders import SiglipImageEmbedder

    proc_cls = getattr(transformers, "SiglipImageProcessorPil", None) or transformers.SiglipImageProcessor
    cfg = transformers.SiglipConfig(
        text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                         vocab_size=100),
        vision_config=dict(hidden_size=48, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                           image_size=32, patch_size=8),
    )
    e = SiglipImageEmbedder("tiny-random", device="cpu", model=transformers.SiglipModel(cfg),
                            processor=proc_cls(size={"height": 32, "width": 32}))
    vecs = e.embed_images([Image.new("RGB", (300, 120), "red"), Image.new("RGB", (64, 64), "green")])
    assert e.dim == 48 and len(vecs) == 2
    assert all(abs(float(np.linalg.norm(v)) - 1) < 1e-4 for v in vecs)


def test_http_image_embedder_contract():
    def handler(req: httpx.Request) -> httpx.Response:
        n = len(json.loads(req.content)["images"])
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0]] * n})

    e = HTTPImageEmbedder("http://encoder/embed", dim=2)
    e.http = httpx.Client(transport=httpx.MockTransport(handler))
    assert e.embed_images([Image.new("RGB", (10, 10))]) == [[pytest.approx(0.6), pytest.approx(0.8)]]


# --------------------------------------------------------------------------- 识别


def test_images_are_indexed(kb):
    st = kb.store.stats()
    assert st["images"] == 22 and st["image_embedding_model"].startswith("handcrafted")


def test_label_photo_is_confirmed_by_ocr_and_serial_decoded(kb):
    res = kb.identify_photo(PHOTOS / "site_310-04521-A_label.jpg")
    assert res.identified and res.top.part_number == "310-04521-A" and res.top.ocr_confirmed
    assert res.manufacture_date == "2022-01"
    assert any("已被 310-04521-B 替代" in f for f in res.facts)  # 停产件 → 提醒替代件


def test_lookalikes_without_label_ask_for_confirmation(kb):
    res = kb.identify_photo(PHOTOS / "site_220-11008-A_nolabel.jpg")
    assert {res.top.entity_id, *res.lookalikes} >= {"part:220-11008-A", "part:220-11009-A"}
    assert any("外观几乎相同" in w for w in res.warnings)
    fans = kb.identify_photo(PHOTOS / "site_310-04521-B_nolabel.jpg")
    assert "part:310-04521-A" in fans.lookalikes  # 新旧版本：按物料主数据的替代关系放宽提醒


def test_unknown_object_is_rejected(kb):
    res = kb.identify_photo(PHOTOS / "site_unknown_helmet.jpg")
    assert not res.identified and res.top is None and res.hint_entities() == []
    assert any("未能可靠识别" in w for w in res.warnings)


def test_small_part_in_high_res_photo(kb):
    big = Image.new("RGB", (3600, 3600), (120, 122, 125))
    big.paste(Image.open(PHOTOS / "site_310-04522-A_nolabel.jpg"), (2700, 2900))
    res = kb.identify_photo(big)
    assert res.identified and res.top.part_number == "310-04522-A"


def test_ask_with_photo_feeds_object_layer_and_kb(kb):
    ident, ans = kb.ask_with_photo(PHOTOS / "site_310-04521-A_label.jpg", "怎么更换？")
    assert ident.hint_entities() == ["part:310-04521-A"]
    assert any("已被 310-04521-B 替代" in f for f in ans.facts)
    assert any("出厂年月为 2022-01" in f for f in ans.facts)
    assert not ans.escalate and any(c.doc_ref.startswith(("manuals:", "bulletins:")) for c in ans.citations)


def test_photo_eval_has_no_confident_wrong_answers(kb):
    from servicekb.vision.evaluate import run_photo_eval

    rep = run_photo_eval(kb, PHOTOS / "labels.jsonl")
    assert rep.metrics["top1"] >= 0.95 and rep.metrics["confident_wrong"] == 0.0
    vis = run_photo_eval(kb, PHOTOS / "labels.jsonl", use_ocr=False)
    assert vis.metrics["confident_wrong"] == 0.0  # 只看外观时可以认不出，但不能自信地认错


# --------------------------------------------------------------------------- 增量入库


def test_image_ingest_is_incremental(tmp_path, fresh_kb):
    from servicekb.connectors.part_images import PartImageConnector

    lib = tmp_path / "lib"
    for pn in ("310-04521-A", "520-00310-A"):
        shutil.copytree(REFS / pn, lib / pn)
    conn = PartImageConnector("imgs", lib)
    r1 = fresh_kb.pipeline.run(conn)
    assert r1.images_indexed == 4
    r2 = fresh_kb.pipeline.run(conn)
    assert (r2.images_indexed, r2.images_unchanged) == (0, 4)
    (lib / "520-00310-A" / "front.png").unlink()
    r3 = fresh_kb.pipeline.run(conn)
    assert (r3.images_deleted, r3.images_unchanged) == (1, 3)


# --------------------------------------------------------------------------- API


def test_photo_api(client):
    h = {"X-API-Key": "secret-key", "X-User-Groups": "public,service"}
    with (PHOTOS / "site_220-11009-A_label.jpg").open("rb") as f:
        r = client.post("/v1/photo/identify", files={"image": ("p.jpg", f, "image/jpeg")}, headers=h)
    body = r.json()
    assert r.status_code == 200 and body["identified"] and body["candidates"][0]["part_number"] == "220-11009-A"
    with (PHOTOS / "site_220-11009-A_label.jpg").open("rb") as f:
        r = client.post("/v1/photo/ask", files={"image": ("p.jpg", f, "image/jpeg")},
                        data={"question": "这个开关能用在 AS-250K-HV 上吗？"}, headers=h)
    assert r.status_code == 200 and r.json()["answer"]["citations"]
    r = client.post("/v1/photo/identify", files={"image": ("x.jpg", b"not an image", "image/jpeg")}, headers=h)
    assert r.status_code == 422


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from servicekb.api import create_app
    from tests.conftest import memory_settings

    with TestClient(create_app(memory_settings(api_keys=["secret-key"]))) as c:
        yield c
