function stack_frame(newData, data) {
    if (data.length === 0) {
        data = Array(65).fill(0);
    }
    data = data.concat(newData);
    return data.slice(13);
}

class PPO {
    constructor() {
        this.model = null;
        this.load_model();
        this.loaded = false;
    }
    async load_model() {
        const option = { executionProviders: ["webgl"] };
        this.model = await ort.InferenceSession.create(
            "static/model/flappy_mlp.onnx",
            option
        );
        console.log("model loaded");
        this.loaded = true;
    }
    async predict(obs) {
        const input = {
            input: obs,
        };
        const results = await this.model.run(input);
        return results.output.data;
    }
}

class Random {
    constructor(seed = 88675123) {
        this.x = 123456789;
        this.y = 362436069;
        this.z = 521288629;
        this.w = seed;
    }

    next() {
        let t;

        t = this.x ^ (this.x << 11);
        this.x = this.y;
        this.y = this.z;
        this.z = this.w;
        this.w = this.w ^ (this.w >> 19) ^ (t ^ (t >> 8));
        return this.w;
    }
    nextInt(min, max) {
        const r = Math.abs(this.next());
        return min + (r % (max + 1 - min));
    }
}
