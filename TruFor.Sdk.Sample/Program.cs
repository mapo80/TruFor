using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using TruFor.Sdk;

if (args.Length < 2)
{
    Console.WriteLine("Usage: TruFor.Sdk.Sample <imagePath> <outputDir>");
    return;
}

var imagePath = args[0];
var outputDir = args[1];
Directory.CreateDirectory(outputDir);

using var bitmap = SKBitmap.Decode(imagePath);
using var sdk = new TruForSdk();

var np = sdk.RunNoiseprint(bitmap);
File.WriteAllText(Path.Combine(outputDir, "noiseprint.json"), TensorToJson(np));

var segs = sdk.RunSegmentation(bitmap);
for (int i = 0; i < segs.Count; i++)
{
    File.WriteAllText(Path.Combine(outputDir, $"seg_{i}.json"), TensorToJson(segs[i]));
}

static string TensorToJson(DenseTensor<float> t)
{
    var obj = new { shape = t.Dimensions.ToArray(), data = t.Buffer.ToArray() };
    return JsonSerializer.Serialize(obj);
}
