using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace TruFor.Sdk;

public sealed class TruForSdk : IDisposable
{
    private readonly InferenceSession _noiseprint;
    private readonly InferenceSession _mit;

    public TruForSdk(string? modelsDirectory = null)
    {
        modelsDirectory ??= Path.Combine(AppContext.BaseDirectory, "models");

        if (!Directory.Exists(modelsDirectory))
            throw new DirectoryNotFoundException(modelsDirectory);

        var noiseprintPath = Path.Combine(modelsDirectory, "noiseprint_pp.onnx");
        if (!File.Exists(noiseprintPath))
            throw new FileNotFoundException(noiseprintPath);
        _noiseprint = new InferenceSession(noiseprintPath);

        var mitPath = Path.Combine(modelsDirectory, "mit_b2.onnx");
        if (!File.Exists(mitPath))
            throw new FileNotFoundException(mitPath);
        _mit = new InferenceSession(mitPath);
    }

    private static DenseTensor<float> BitmapToTensor(SKBitmap bmp)
    {
        var tensor = new DenseTensor<float>(new[] { 1, 3, bmp.Height, bmp.Width });
        for (int y = 0; y < bmp.Height; y++)
        {
            for (int x = 0; x < bmp.Width; x++)
            {
                var c = bmp.GetPixel(x, y);
                tensor[0, 0, y, x] = c.Red / 255f;
                tensor[0, 1, y, x] = c.Green / 255f;
                tensor[0, 2, y, x] = c.Blue / 255f;
            }
        }
        return tensor;
    }

    public DenseTensor<float> RunNoiseprint(SKBitmap bitmap)
    {
        var input = BitmapToTensor(bitmap);
        var inputs = new[] { NamedOnnxValue.CreateFromTensor("image", input) };
        using var results = _noiseprint.Run(inputs);
        return (DenseTensor<float>)results.First().Value;
    }

    public IReadOnlyList<DenseTensor<float>> RunSegmentation(SKBitmap bitmap)
    {
        var rgb = BitmapToTensor(bitmap);
        var np = RunNoiseprint(bitmap);
        var np3 = new DenseTensor<float>(new[] { 1, 3, bitmap.Height, bitmap.Width });
        for (int y = 0; y < bitmap.Height; y++)
        {
            for (int x = 0; x < bitmap.Width; x++)
            {
                float v = np[0, 0, y, x];
                np3[0, 0, y, x] = v;
                np3[0, 1, y, x] = v;
                np3[0, 2, y, x] = v;
            }
        }

        var inputs = new[]
        {
            NamedOnnxValue.CreateFromTensor("rgb", rgb),
            NamedOnnxValue.CreateFromTensor("np", np3)
        };
        using var results = _mit.Run(inputs);
        return results.Select(r => (DenseTensor<float>)r.Value).ToList();
    }

    public void Dispose()
    {
        _noiseprint.Dispose();
        _mit.Dispose();
    }
}
