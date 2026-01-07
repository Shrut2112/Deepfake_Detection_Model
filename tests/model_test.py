from app.model import Model

def test_model_loads():
    model = Model()
    assert model.model is not None