using System.IO.MemoryMappedFiles;
using System.Text.Json;
using LK_FrameGrabber;
using OpenCvSharp;

internal static class Program
{
    private const int HeaderSize = 48;
    private const int MaximumFrameBytes = 4 * 1024 * 1024;
    private static readonly object FrameLock = new();
    private static MemoryMappedFile? sharedMemory;
    private static LK_Library.Marshal? camera;
    private static long sequence;
    private static long frameNumber;
    private static int expectedPixelBytes;
    private static bool stopping;
    private static bool frameFormatErrorReported;

    private static int Main(string[] args)
    {
        try
        {
            var options = ParseArguments(args);
            string mapName = Required(options, "map-name");
            string camFile = Required(options, "cam-file");
            string connector = options.GetValueOrDefault("connector", "B").ToUpperInvariant();
            string pixelFormat = options.GetValueOrDefault("pixel-format", "Mono16");
            expectedPixelBytes = pixelFormat.Equals("Mono8", StringComparison.OrdinalIgnoreCase) ? 1 : 2;

            if (!File.Exists(camFile))
                throw new FileNotFoundException("CAM 파일을 찾을 수 없습니다.", camFile);
            if (connector is not ("A" or "B"))
                throw new ArgumentException("커넥터는 A 또는 B여야 합니다.");

            sharedMemory = MemoryMappedFile.CreateOrOpen(
                mapName, HeaderSize + MaximumFrameBytes, MemoryMappedFileAccess.ReadWrite);
            camera = new LK_Library.Marshal();
            camera.EventCameraCallback += OnCameraFrame;

            var board = connector == "A" ? eBoardConnector.A : eBoardConnector.B;
            (bool connected, string cameraMessage, string serialMessage) =
                camera.CameraConnect(eCameraInterface.FrameGrabber, -1, board, camFile);
            if (!connected)
                throw new InvalidOperationException($"카메라 연결 실패: {cameraMessage} {serialMessage}".Trim());

            camera.SetLive();
            camera.SourceActive();
            WriteStatus("connected", cameraMessage, serialMessage);

            string? command;
            while (!stopping && (command = Console.ReadLine()) is not null)
            {
                if (command.Equals("quit", StringComparison.OrdinalIgnoreCase))
                    break;
                if (command.Equals("status", StringComparison.OrdinalIgnoreCase))
                    WriteStatus("connected", "", "");
            }
            return 0;
        }
        catch (Exception exc)
        {
            WriteStatus("error", exc.Message, "");
            return 1;
        }
        finally
        {
            stopping = true;
            if (camera is not null)
            {
                try { camera.EventCameraCallback -= OnCameraFrame; } catch { }
                try { camera.CameraDisconnect(); } catch { }
            }
            sharedMemory?.Dispose();
        }
    }

    private static void OnCameraFrame(Mat frame, double fps, string channelState)
    {
        if (stopping || sharedMemory is null || frame is null || frame.Empty()) return;
        try
        {
            byte[] data;
            int pixelBytes;
            if (frame.Type() == MatType.CV_16UC1)
            {
                frame.GetArray(out ushort[] pixels);
                data = new byte[pixels.Length * sizeof(ushort)];
                Buffer.BlockCopy(pixels, 0, data, 0, data.Length);
                pixelBytes = 2;
            }
            else if (frame.Type() == MatType.CV_8UC1)
            {
                frame.GetArray(out data);
                pixelBytes = 1;
            }
            else
            {
                WriteStatus("frame_error", $"지원하지 않는 프레임 형식: {frame.Type()}", "");
                return;
            }
            if (pixelBytes != expectedPixelBytes)
            {
                if (!frameFormatErrorReported)
                {
                    frameFormatErrorReported = true;
                    WriteStatus("frame_error", "설정한 픽셀 형식과 수신 프레임 형식이 다릅니다.", "");
                }
                return;
            }
            if (data.Length > MaximumFrameBytes) return;

            lock (FrameLock)
            {
                sequence += 2;
                using var accessor = sharedMemory.CreateViewAccessor();
                accessor.Write(8, sequence - 1); // odd: writing
                accessor.Write(0, 0x53535054);   // SSPT
                accessor.Write(4, 1);
                accessor.Write(16, ++frameNumber);
                accessor.Write(24, frame.Width);
                accessor.Write(28, frame.Height);
                accessor.Write(32, pixelBytes);
                accessor.Write(36, data.Length);
                accessor.Write(40, (int)Math.Round(fps * 1000.0));
                accessor.Write(44, ChannelStateCode(channelState));
                accessor.WriteArray(HeaderSize, data, 0, data.Length);
                accessor.Write(8, sequence);     // even: complete
                accessor.Flush();
            }
        }
        catch (Exception exc)
        {
            WriteStatus("frame_error", exc.Message, "");
        }
    }

    private static int ChannelStateCode(string state) => state?.ToUpperInvariant() switch
    {
        "ACTIVE" => 2,
        "READY" => 1,
        _ => 0,
    };

    private static Dictionary<string, string> ParseArguments(string[] args)
    {
        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (int index = 0; index + 1 < args.Length; index += 2)
            values[args[index].TrimStart('-')] = args[index + 1];
        return values;
    }

    private static string Required(Dictionary<string, string> values, string key) =>
        values.TryGetValue(key, out string? value) && !string.IsNullOrWhiteSpace(value)
            ? value : throw new ArgumentException($"필수 인자가 없습니다: --{key}");

    private static void WriteStatus(string status, string message, string detail)
    {
        Console.WriteLine(JsonSerializer.Serialize(new { status, message, detail }));
        Console.Out.Flush();
    }
}
