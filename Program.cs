using System.Drawing;
using System;

struct img
{
    public List<double> x;
    public int y;
}
class program
{
    static private List<img> trainData = new List<img>();
    static List<double> hogFeatures(int[,] arr)
    {
        int[,] xGra = new int[28,28];
        int[,] yGra = new int[28, 28];
        double[,] gra = new double[28, 28];
        double[,] degree = new double[28, 28];
        List<List<double>> features = new List<List<double>>();
        for (int i = 0; i < 28; i++)
        {
            for (int j = 0;j < 28; j++)
            {
                if (j == 0)
                {
                    xGra[i, j] = arr[i, j + 1];
                }else if (j == 27)
                {
                    xGra[i, j] = arr[i, j - 1];
                }
                else
                {
                    xGra[i, j] = arr[i, j + 1] - arr[i, j - 1];
                }
            }
        }

        for (int i = 0; i < 28; i++)
        {
            for (int j = 0; j < 28; j++)
            {
                if (i == 0)
                {
                    yGra[i, j] = arr[i + 1, j];
                }
                else if (i == 27)
                {
                    yGra[i, j] = arr[i - 1, j];
                }
                else
                {
                    yGra[i, j] = arr[i + 1, j] - arr[i - 1, j];
                }
            }

        }

        for (int i = 0; i < 28; i++)
        {
            for (int j = 0; j < 28; j++)
            {
                gra[i,j] = Math.Sqrt((double)xGra[i,j]*(double)xGra[i, j] + (double)yGra[i, j]*(double)yGra[i, j]);
                degree[i, j] =Math.Abs(Math.Atan((double)yGra[i, j] / (double)xGra[i, j])) * 180 / Math.PI;
            }
        }
        for (int l = 0; l < 28; l+=7)
        {
            for (int k = 0; k < 28; k += 7)
            {
                List<double> feature = new List<double>() { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
                for (int i = 0 + l; i < 7 + l; i++)
                {
                    for (int j = 0 + k; j < 7 + k; j++)
                    {
                        if (degree[i, j] >= 0 && degree[i, j] <= 20)
                        {
                            feature[0] += gra[i, j];
                        }
                        else if (degree[i, j] >= 20 && degree[i, j] < 40)
                        {
                            feature[1] += gra[i, j];
                        }
                        else if (degree[i, j] >= 40 && degree[i, j] < 60)
                        {
                            feature[2] += gra[i, j];
                        }
                        else if (degree[i, j] >= 60 && degree[i, j] < 80)
                        {
                            feature[3] += gra[i, j];
                        }
                        else if (degree[i, j] >= 80 && degree[i, j] < 100)
                        {
                            feature[4] += gra[i, j];
                        }
                        else if (degree[i, j] >= 100 && degree[i, j] < 120)
                        {
                            feature[5] += gra[i, j];
                        }
                        else if (degree[i, j] >= 120 && degree[i, j] < 140)
                        {
                            feature[6] += gra[i, j];
                        }
                        else if (degree[i, j] >= 140 && degree[i, j] < 160)
                        {
                            feature[7] += gra[i, j];
                        }
                        else
                        {
                            feature[8] += gra[i, j];
                        }
                    }
                }
                features.Add(feature);
            }
        }
        List<double> featuresVector = new List<double>();
        var len = features.Count;
        for (var i = 0; i < len; i++)
        {
            var eleLen = features[i].Count;
            for (var j = 0; j < eleLen; j++)
            {
                featuresVector.Add(features[i][j]);
            }
        }
        return featuresVector;
    }
    static void scaleData()
    {
        var length = trainData.Count;
        for (int i = 0; i < length; i++)
        {
            double soq = 0;
            for (int j = 0;j < 144; j++)
            {
                soq += trainData[i].x[j]* trainData[i].x[j];
            }
            soq = Math.Sqrt(soq);
            for (int j = 0; j < 144; j++)
            {
                trainData[i].x[j] /= soq;
            }
        }
    }
    static void scale(ref img point)
    {
        double soq = 0;
        for (int j = 0; j < 144; j++)
        {
            soq += point.x[j] * point.x[j];
        }
        soq = Math.Sqrt(soq);
        for (int j = 0; j < 144; j++)
        {
            point.x[j] /= soq;
        }
    }
    static int[,] getMatrix(string dir)
    {
        int[,] matrix = new int[28, 28];
        var img = new Bitmap(dir);
        for (int i = 0; i < img.Width; i++)
        {
            for (int j = 0; j < img.Height; j++)
            {
                Color pixel = img.GetPixel(i, j);
                matrix[i, j] = (int)pixel.R;
            }
        }
        return matrix;
    }
    static void settingData()
    {
        string dir0 = Directory.GetCurrentDirectory() + "\\train\\0";
        string dir1 = Directory.GetCurrentDirectory() + "\\train\\1";

        string[] dataset0 = Directory.GetFiles(dir0, "*.*", SearchOption.AllDirectories);
        string[] dataset1 = Directory.GetFiles(dir1, "*.*", SearchOption.AllDirectories);
        foreach (var file in dataset0)
        {
            img img0;
            img0.x = hogFeatures(getMatrix(file));
            img0.y = 0;
            trainData.Add(img0);
        }
        foreach (var file in dataset1)
        {
            img img1;
            img1.x = hogFeatures(getMatrix(file));
            img1.y = 1;
            trainData.Add(img1);
        }
    }

    static double sigmoid(img point,ref List<double> weights,ref double bias)
    {
        double value = 0;
        for (int i = 0; i < 144; i++)
        {
            value += point.x[i] * weights[i];   
        }
        value += bias;
        return 1.0/(1 + Math.Exp(-(value)));
    }
    static double costFunction(ref List<double> weights, ref double bias)
    {
        double result = 0.0;
        var size = trainData.Count;
        for (var i = 0;i < size; i++)
        {
            var value = 0.0;
            for (var j = 0; j< 144; j++)
            {
                value = weights[j] * trainData[i].x[j];
            }
            result += trainData[i].y * Math.Log(1 + Math.Exp(-(value + bias))) + (1 - trainData[i].y) * (value + bias);
        }
        return result;
    }
    
    static void update(ref List<double> weights,ref double bias,double learningRate)
    {
        var size = trainData.Count;
        var length = weights.Count;
        List<double> tempWeights = new List<double>();
        for (var i = 0; i < length; i++)
        {
            tempWeights.Add(0.0);
        }
        double tempBias = 0.0;  
        for (var i = 0;i < size; i++)
        {
            for (var j = 0; j < length; j++)
            {   
                tempWeights[j] += (trainData[i].x[j]*(sigmoid(trainData[i],ref weights,ref bias) - trainData[i].y)) ;
            }
            tempBias += (sigmoid(trainData[i], ref weights, ref bias) - trainData[i].y);
        }   
        for (var i = 0; i < length; i++)
        {
            weights[i] -= learningRate*tempWeights[i]/size;
        }
        bias -= learningRate*tempBias/size;
    }
    static void train(ref List<double> weights, ref double bias, double learningRate,long trainingTime)
    {
        for (var i = 0; i < trainingTime; i++)
        {
            update(ref weights, ref bias, learningRate);
            for (var j = 0;j < 10; j++)
            {
                Console.Write(" " + weights[j]);
            }
            Console.WriteLine(bias);
        }
    }
    static void predict(ref img point,ref List<double> weights,ref double bias)
    {
        double pre = sigmoid(point,ref weights,ref bias);
        Console.WriteLine("-----------------------");
        Console.WriteLine(pre);
        if (pre >= 0.5)
        {
            point.y = 1;
            Console.WriteLine("This is number 1");
        }
        else
        {
            point.y = 0;
            Console.WriteLine("This is number 0");
        }
    }
    static void Main()
    {
        settingData();
        scaleData();
        List<double> weights = new List<double>();
        Random random = new Random();
        for (var i = 0; i < 144; i++)
        {
            weights.Add(random.NextDouble());
        }
        double bias = 0.15;
        train(ref weights, ref bias, 0.1, 500);
        img point = new img();
        int[,] matrix = getMatrix("test.png");
        point.x = hogFeatures(matrix);
        scale(ref point);
        predict(ref point, ref weights, ref bias);
    }
}


