"""Same-observation author-adapter and RNG diagnosis, not a success-rate run."""
import argparse
import ctypes
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from roboeval.policies.smolvla import SmolVLAPolicy as AuthorAdapter
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from smolvla_clean_repro import axis


def digest(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def main(a):
    ck = Path('/hy-tmp/smolvla_ckpt/HuggingFaceVLA_smolvla_libero')
    model = SmolVLAPolicy.from_pretrained(ck).eval().cuda().requires_grad_(False)
    stats = {}
    for filename in ('policy_preprocessor_step_5_normalizer_processor.safetensors',
                     'policy_postprocessor_step_1_unnormalizer_processor.safetensors'):
        with safe_open(ck / filename, framework='pt', device='cpu') as f:
            stats.update({k: f.get_tensor(k) for k in f.keys()})
    for module, key in [(model.normalize_inputs, 'observation.state'),
                        (model.normalize_targets, 'action'), (model.unnormalize_outputs, 'action')]:
        buffers = getattr(module, 'buffer_' + key.replace('.', '_'))
        for kind in ('mean', 'std'):
            buffers[kind].data.copy_(stats[key + '.' + kind].to(buffers[kind]))
    adapter = AuthorAdapter(device='cuda')
    adapter._torch = torch
    adapter._model = model
    suite = benchmark.get_benchmark_dict()['libero_spatial']()
    task = suite.get_task(0)
    states = torch.load(Path(get_libero_path('init_states')) / 'libero_spatial' /
                        (task.name + '.pruned_init'), map_location='cpu', weights_only=False)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    ctypes.CDLL(None).srand(ctypes.c_uint(0))
    env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) /
                            task.problem_folder / task.bddl_file),
                            camera_heights=256, camera_widths=256, render_gpu_device_id=0)
    try:
        env.noise = 0; env.seed(0); env.reset(); obs = env.set_init_state(states[0])
        for _ in range(10):
            obs = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))[0]
        for robot in env.robots:
            robot.controller.use_delta = True
        obs = dict(obs, language=task.language)
        state = np.concatenate([obs['robot0_eef_pos'], axis(obs['robot0_eef_quat']),
                                obs['robot0_gripper_qpos']]).astype(np.float32)
        def image(k):
            return torch.from_numpy(np.ascontiguousarray(np.asarray(obs[k])[::-1, ::-1])).permute(2,0,1)[None].float().cuda()/255.0
        ours = {'observation.images.image': image('agentview_image'),
                'observation.images.image2': image('robot0_eye_in_hand_image'),
                'observation.state': torch.from_numpy(state)[None].cuda(), 'task': task.language}
        author = adapter._build_observation(obs)
        inputs = {}
        for k in ours:
            if torch.is_tensor(ours[k]):
                inputs[k] = {'exact_equal': torch.equal(ours[k], author[k]),
                             'max_abs_diff': float((ours[k] - author[k]).abs().max()),
                             'shape': list(ours[k].shape)}
            else:
                inputs[k] = {'exact_equal': ours[k] == author[k]}
        def predict(x):
            model.reset(); torch.manual_seed(17); torch.cuda.manual_seed_all(17)
            with torch.inference_mode():
                action = model.select_action(x).detach().cpu().numpy().reshape(-1)
            return np.clip(action, -1, 1).astype(np.float32)
        ours_action = predict(ours); author_action = predict(author)
        ours_repeat = predict(ours); author_repeat = predict(author)
        aligned = dict(ours)
        for key, camera in [('observation.images.image', 'agentview_image'),
                            ('observation.images.image2', 'robot0_eye_in_hand_image')]:
            x = np.ascontiguousarray(obs[camera][::-1, ::-1]).astype(np.float32) / 255.0
            aligned[key] = torch.from_numpy(x).permute(2, 0, 1)[None].cuda()
        aligned_action = predict(aligned)
        def author_action_once():
            adapter.reset(77)
            return adapter.act(obs)
        # Author reset takes a seed argument but does not restore Torch's RNG.
        torch.manual_seed(123); torch.cuda.manual_seed_all(123)
        rng_before = digest(torch.cuda.get_rng_state())
        unseeded_a = author_action_once()
        rng_after_a = digest(torch.cuda.get_rng_state())
        unseeded_b = author_action_once()
        rng_after_b = digest(torch.cuda.get_rng_state())
        seeded = []
        for _ in range(2):
            torch.manual_seed(123); torch.cuda.manual_seed_all(123)
            seeded.append(author_action_once())
        result = {'scope': 'author observation/action adapter using same already-loaded model; loader not re-executed',
                  'task': task.name, 'init_index': 0,
                  'input_comparison': inputs,
                  'fixed_rng_action_comparison': {'ours': ours_action.tolist(), 'author': author_action.tolist(),
                        'exact_equal': bool(np.array_equal(ours_action, author_action)),
                        'max_abs_diff': float(np.max(np.abs(ours_action-author_action)))},
                  'repeat_fixed_rng': {'ours_exact_equal': bool(np.array_equal(ours_action, ours_repeat)),
                                       'author_exact_equal': bool(np.array_equal(author_action, author_repeat))},
                  'cpu_aligned': {'all_inputs_exact_equal': all(torch.equal(aligned[k], author[k]) if torch.is_tensor(aligned[k]) else aligned[k] == author[k] for k in aligned),
                                  'action_exact_equal': bool(np.array_equal(aligned_action, author_action)),
                                  'action_max_abs_diff': float(np.max(np.abs(aligned_action-author_action)))},
                  'bfloat16_parameter_examples': [name for name, p in model.named_parameters() if p.dtype == torch.bfloat16][:8],
                  'author_reset_same_seed_without_torch_reseed': {
                        'action_a': unseeded_a.tolist(), 'action_b': unseeded_b.tolist(),
                        'exact_equal': bool(np.array_equal(unseeded_a, unseeded_b)),
                        'difference_l2': float(np.linalg.norm(unseeded_a-unseeded_b)),
                        'cuda_rng_hashes': [rng_before, rng_after_a, rng_after_b]},
                  'explicit_torch_reseed_action_exact_equal': bool(np.array_equal(*seeded)),
                  'all_floating_parameter_dtypes': sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()})}
        assert result['explicit_torch_reseed_action_exact_equal']
        assert not a.out.exists()
        a.out.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
