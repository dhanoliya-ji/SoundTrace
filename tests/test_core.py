import torch

from soundtrace.cli import avsbench_f_measure, f_measure_curves, metric
from soundtrace.data import audio_window, shift_log_mel, synthetic_batch, validate_manifest
from soundtrace.model import SoundTraceNet


def test_audio_window_center_and_padding():
    wave=torch.arange(10,dtype=torch.float32)
    assert audio_window(wave, 5, 1, 4).tolist()==[3,4,5,6]
    assert audio_window(wave, 1, 1, 4).tolist()==[0,0,1,2]


def test_log_mel_time_shift_zero_pads_without_wrapping():
    feature=torch.arange(12,dtype=torch.float32).reshape(1,6,2)
    shifted=shift_log_mel(feature,1/3,1.0)
    assert shifted.tolist()==[[[4,5],[6,7],[8,9],[10,11],[0,0],[0,0]]]
    reverse=shift_log_mel(feature,-1/3,1.0)
    assert reverse.tolist()==[[[0,0],[0,0],[0,1],[2,3],[4,5],[6,7]]]

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


def test_both_model_variants_accept_official_log_mel_features():
    video=torch.rand(2,3,32,32); audio=torch.rand(2,1,96,64)
    for gated in (False,True):
        output=SoundTraceNet(8,gated)(video,audio)
        assert output['logits'].shape==(2,1,32,32)

def test_metric_perfect():
    target=torch.zeros(1,1,8,8); target[:,:,2:5,2:5]=1
    assert metric((target*20)-10,target)==1


def test_metric_uses_official_empty_mask_iou_convention():
    empty=torch.zeros(1,1,2,2)
    false_positive=torch.tensor([[[[20.0,-20.0],[-20.0,-20.0]]]])
    assert metric(torch.full_like(empty,-20),empty)==1
    assert metric(false_positive,empty)==0.75


def test_avsbench_f_measure_uses_threshold_curve_and_skips_empty_masks():
    target=torch.zeros(2,1,4,4);target[0,0,1:3,1:3]=1
    logits=torch.where(target>0,torch.tensor(20.0),torch.tensor(-20.0))
    curves=f_measure_curves(logits,target)
    score=avsbench_f_measure({'video-a':[curves[0]],'empty-video':[curves[1]]})
    assert score is not None and score>0.999

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
    import pickle

    from PIL import Image

    from soundtrace.data import (
        build_avsbench_manifest,
        load_record,
        read_manifest,
        validate_manifest,
    )
    subset=tmp_path/'Single-source'; data=subset/'s4_data'; meta=subset/'s4_meta_data.csv'
    subset.mkdir(parents=True)
    row={'video_id':'clipA','start_time':'0','category':'guitar','split':'train'}
    with meta.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=row.keys());writer.writeheader();writer.writerow(row)
    frame=data/'visual_frames'/'train'/'guitar'/'clipA';frame.mkdir(parents=True)
    masks=data/'gt_masks'/'train'/'guitar'/'clipA';masks.mkdir(parents=True)
    for i in range(1,6):
        Image.new('RGB',(8,8)).save(frame/f'clipA_{i}.png')
        Image.new('L',(8,8),255).save(masks/f'clipA_{i}.png')
    features=data/'audio_log_mel'/'train'/'guitar';features.mkdir(parents=True)
    with (features/'clipA.pkl').open('wb') as handle:
        pickle.dump(torch.zeros(5,1,96,64),handle)
    output=tmp_path/'s4.jsonl'; result=build_avsbench_manifest(tmp_path,output,'s4')
    records=read_manifest(output)
    assert result['valid'] and len(records)==1
    assert records[0]['annotation_frame']==1 and records[0]['timestamp']==.5
    assert validate_manifest(records,output)['valid']
    image,audio,mask=load_record(records[0],output,16,16000,1.0)
    assert image.shape==(3,16,16) and audio.shape==(1,96,64) and mask.shape==(1,16,16)


def test_official_avsbench_ms3_manifest_uses_video_frame_names(tmp_path):
    import csv
    import pickle

    from PIL import Image

    from soundtrace.data import build_avsbench_manifest, read_manifest, validate_manifest
    subset=tmp_path/'Multi-sources'; data=subset/'ms3_data'; meta=subset/'ms3_meta_data.csv'
    subset.mkdir(parents=True)
    with meta.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=['video_name','split'])
        writer.writeheader();writer.writerow({'video_name':'clipB','split':'train'})
    frames=data/'visual_frames'/'clipB';frames.mkdir(parents=True)
    masks=data/'gt_masks'/'train'/'clipB';masks.mkdir(parents=True)
    for index in range(1,6):
        Image.new('RGB',(8,8)).save(frames/f'clipB.mp4_{index}.png')
        Image.new('L',(8,8),255).save(masks/f'clipB_{index}.png')
    features=data/'audio_log_mel'/'train';features.mkdir(parents=True)
    with (features/'clipB.pkl').open('wb') as handle:
        pickle.dump(torch.zeros(5,1,96,64),handle)
    output=tmp_path/'ms3.jsonl';result=build_avsbench_manifest(tmp_path,output,'ms3')
    records=read_manifest(output)
    assert result['valid'] and len(records)==5
    assert validate_manifest(records,output)['valid']
    assert records[0]['audio_index']==0 and records[-1]['audio_index']==4

def test_decode_audio_reports_missing_ffmpeg_or_track(tmp_path):
    import pytest

    from soundtrace.data import decode_audio
    with pytest.raises((RuntimeError, ValueError)):
        decode_audio(tmp_path/'not_a_media_file.mp4')
