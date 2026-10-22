import torch
from soundtrace.data import audio_window, synthetic_batch, validate_manifest
from soundtrace.model import SoundTraceNet
from soundtrace.cli import metric

def test_audio_window_center_and_padding():
    wave=torch.arange(10,dtype=torch.float32)
    assert audio_window(wave, 5, 1, 4).tolist()==[3,4,5,6]
    assert audio_window(wave, 1, 1, 4).tolist()==[0,0,1,2]

def test_both_model_variants_shapes_and_gate_gradients():
    video,audio,mask=synthetic_batch(2,32,512)
    for gated in (False,True):
        model=SoundTraceNet(8,gated)
        out=model(video,audio)
        assert out['logits'].shape==mask.shape
        assert out['gate'].shape==(2,16)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(out['logits'],mask)
        loss.backward()
        if gated: assert model.gate[-2].weight.grad is not None

def test_metric_perfect():
    target=torch.zeros(1,1,8,8); target[:,:,2:5,2:5]=1
    assert metric((target*20)-10,target)==1

def test_manifest_split_leakage_and_missing_files(tmp_path):
    rows=[{"video":"v.mp4","mask":"m.png","video_id":"same","timestamp":0,"split":"train"},
          {"video":"v.mp4","mask":"m.png","video_id":"same","timestamp":1,"split":"test"}]
    result=validate_manifest(rows,tmp_path/'manifest.jsonl')
    assert not result['valid']
    assert any('leakage' in e['error'] for e in result['errors'])
    assert any('video missing' in e['error'] for e in result['errors'])

def test_manifest_rejects_unreviewed_discovery_and_empty_manifest(tmp_path):
    assert not validate_manifest([],tmp_path/'empty.jsonl')['valid']
    row={"video":"v.mp4","mask":"m.png","video_id":"v","timestamp":0,"review_required":True}
    result=validate_manifest([row],tmp_path/'manifest.jsonl')
    assert any('manually reviewed' in e['error'] for e in result['errors'])

def test_official_avsbench_s4_manifest_obeys_one_shot_train_labels(tmp_path):
    import csv
    from soundtrace.data import build_avsbench_manifest, read_manifest
    subset=tmp_path/'Single-source'; data=subset/'s4_data'; meta=subset/'s4_meta_data.csv'
    subset.mkdir(parents=True)
    row={'video_id':'clipA','start_time':'0','category':'guitar','split':'train'}
    with meta.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=row.keys());writer.writeheader();writer.writerow(row)
    (data/'raw_videos'/'train'/'guitar').mkdir(parents=True)
    (data/'raw_videos'/'train'/'guitar'/'clipA.mp4').write_bytes(b'fixture')
    frame=data/'visual_frames'/'train'/'guitar'/'clipA';frame.mkdir(parents=True)
    masks=data/'gt_masks'/'train'/'clipA';masks.mkdir(parents=True)
    for i in range(1,6):
        (frame/f'clipA_{i}.png').write_bytes(b'frame');(masks/f'clipA_{i}.png').write_bytes(b'mask')
    output=tmp_path/'s4.jsonl'; result=build_avsbench_manifest(tmp_path,output,'s4')
    records=read_manifest(output)
    assert result['valid'] and len(records)==1
    assert records[0]['annotation_frame']==1 and records[0]['timestamp']==.5

def test_decode_audio_reports_missing_ffmpeg_or_track(tmp_path):
    from soundtrace.data import decode_audio
    import pytest
    with pytest.raises((RuntimeError, ValueError)):
        decode_audio(tmp_path/'not_a_media_file.mp4')
