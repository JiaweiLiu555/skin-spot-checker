import torch

from src.model import build_contour_model, build_model


def test_contour_model_preserves_rgb_output_before_training():
    source = build_model(pretrained=False).eval()
    contour = build_contour_model(pretrained=False, source_model=source).eval()
    image = torch.randn(2, 3, 96, 96)

    with torch.inference_mode():
        expected = source(image)
        actual = contour(image)

    assert actual.shape == (2, 2)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_sobel_stream_responds_to_a_visible_boundary():
    model = build_contour_model(pretrained=False).eval()
    flat = torch.zeros(1, 3, 64, 64)
    boundary = flat.clone()
    boundary[:, :, :, 32:] = 1.0

    with torch.inference_mode():
        flat_edges = model.contour_map(flat)
        boundary_edges = model.contour_map(boundary)

    assert boundary_edges.mean() > flat_edges.mean()
