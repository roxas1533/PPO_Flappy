function resize(ctx) {
    const canvasInvisible = document.createElement("canvas");
    canvasInvisible.width = 400;
    canvasInvisible.height = 400;
    const ctx2 = canvasInvisible.getContext("2d");
    ctx2.scale(-1, 1);
    ctx2.scale(0.64, 0.64);
    ctx2.drawImage(ctx, -400, 0);
    return ctx2.getImageData(0, 0, 256, 256);
}

function gray(imageData) {
    const obs = [];
    for (let i = 0; i < imageData.data.length; i += 4) {
        const y =
            0.2126 * imageData.data[i] +
            0.7152 * imageData.data[i + 1] +
            0.0722 * imageData.data[i + 2];
        obs[i / 4] = y / 255;
    }
    return obs;
}

function stack_frame(newData, data) {
    let cp_data = [];
    if (data.length === 0) {
        for (let i = 0; i < newData.length; i++) {
            data.push(newData[i]);
            data.push(newData[i]);
            data.push(newData[i]);
            data.push(newData[i]);
            data.push(newData[i]);
        }
        cp_data = data;
    } else {
        for (let i = 0; i < data.length; i += 5) {
            cp_data.push(data[i + 1]);
            cp_data.push(data[i + 2]);
            cp_data.push(data[i + 3]);
            cp_data.push(data[i + 4]);
            cp_data.push(newData[i / 5]);
        }
        // console.log(cp_data.length);
    }
    return cp_data;
}

class PPO {
    constructor() {
        this.model = null;
        this.load_model();
        this.loaded = false;
    }
    async load_model() {
        this.model = await ort.InferenceSession.create(
            "static/model/flappy.onnx"
        );
        console.log("model loaded");
        this.loaded = true;
    }
    async predict(obs) {
        const input = {
            input: obs,
        };
        const results = await this.model.run(input);
        // console.log();
        return results.output.data[0];
    }
}
