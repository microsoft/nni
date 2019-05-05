using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Globalization;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using System.IO;
using RestSharp;
using Microsoft.Aether.Library;
using Microsoft.Aether.DataContracts;
using Newtonsoft.Json;

namespace AetherClient
{
    
    class NNITrialInfo {
        public class NNIJobForm
        {
            public string jobType { get; set; }
            public class hParamRaw
            {
                public string value { get; set; }
                public int index { get; set; }
            }
            public hParamRaw hyperParameters { get; set; }
        }

        public class NNIAetherConfig
        {
            public string codeDir { get; set; }
            public string baseGraph { get; set; }
            public string outputNodeAlias { get; set; }
            public string outputName { get; set; }
        }

        public NNIAetherConfig aetherConfig { get; set; }
        public NNIJobForm form { get; set; }
        public string workingDirectory { get; set; }

    }

    class NNIHparam {
        public int parameter_id { get; set; }
        public string parameter_source { get; set; }
        public Dictionary<string, string> parameters { get; set; }
    }

    class GuidBody
    {   
        public string guid { get; set; }
    }

    class StatusBody
    {
        public string status { get; set; }
        public StatusBody(string status)
        {
            this.status = status;

        }
    }

    class RewardBody
    {
        public float[] reward { get; set; }
    }

    class ClientRunner
    {
        private AetherEnvironment environment;
        private RestClient client;
        private string expId;
        private string trialId;
        private ExperimentStatusCode status;
        private string experimentId = null;
        private readonly Dictionary<ExperimentStatusCode, StatusBody> statusMap = new Dictionary<ExperimentStatusCode, StatusBody>
        {
            { ExperimentStatusCode.NotStarted, new StatusBody("WAITING") },
            { ExperimentStatusCode.Running, new StatusBody("RUNNING") },
            { ExperimentStatusCode.Failed, new StatusBody("FAILED") },
            { ExperimentStatusCode.Canceled, new StatusBody("USER_CANCELED") },
            { ExperimentStatusCode.Finished, new StatusBody("SUCCEEDED") }
        };
        public ClientRunner(string restURL, string expId, string trialId) {
            this.environment = AetherEnvironment.Create();
            this.client = new RestClient(restURL);
            this.expId = expId;
            this.trialId = trialId;
        }

        public async Task runAsync()
        {
            #region submitTrialJob
            var request = new RestRequest(String.Format("/api/v1/nni-aether/trial-meta/{0}/{1}", this.expId, this.trialId), DataFormat.Json);
            var response = this.client.Get(request);

            var serializer = new RestSharp.Serialization.Json.JsonSerializer();

            NNITrialInfo info = serializer.Deserialize<NNITrialInfo>(response);

            IVisualGraph graph = environment.DeserializeGraph(File.ReadAllText(info.aetherConfig.baseGraph));
            NNIHparam hparams = JsonConvert.DeserializeObject<NNIHparam>(info.form.hyperParameters.value);
            foreach (var item in hparams.parameters)
            {
                graph.Parameters.First(e => e.Name == item.Key).Value = item.Value;
            }

            this.experimentId = environment.SubmitExperiment(new ExperimentCreationInfo { Description = "NNI Test" }, graph);

            GuidBody body = new GuidBody { guid = this.experimentId };

            var reqGuid = new RestRequest(String.Format("/api/v1/nni-aether/update-guid/{0}/{1}", this.expId, this.trialId), Method.POST, DataFormat.Json);
            reqGuid.AddJsonBody(body);
            var resGuid = this.client.Execute(reqGuid);
            Console.WriteLine(resGuid.Content);
            #endregion


            #region updateTrialStatus
            while (true)
            {
                ExperimentStatusCode statusTmp = environment.GetExperiment(this.experimentId).GetStatus();
                if (this.status == statusTmp)
                {
                    continue;
                }
                else
                {
                    this.status = statusTmp;
                }
                StatusBody statBody = statusMap[status];
                var reqStatus = new RestRequest(String.Format("/api/v1/nni-aether/update-status/{0}/{1}", this.expId, this.trialId), Method.POST, DataFormat.Json);
                reqStatus.AddJsonBody(statBody);
                var resStatus = client.Execute(reqStatus);

                if (this.status == ExperimentStatusCode.Finished || this.status == ExperimentStatusCode.Failed || this.status == ExperimentStatusCode.Canceled)
                    break;

                await Task.Delay(10000);
            }


            #endregion

            if (this.status == ExperimentStatusCode.Finished)
            {
                Console.WriteLine(String.Format("Aether Job {0} finished", this.experimentId));
                INodeExecution nodeExec = environment.GetExperiment(this.experimentId).GetExecutionGraph().GetNodeExecution(info.aetherConfig.outputNodeAlias);

                string outputfile = String.Format("{0}\\aether_output.txt", info.workingDirectory);
                nodeExec.GetOutput(info.aetherConfig.outputName).Download(outputfile);
                using (StreamReader reader = new StreamReader(outputfile))
                {
                    string line = reader.ReadToEnd();
                    RewardBody rewardBody = new RewardBody { reward = new float[1] { float.Parse(line, CultureInfo.InvariantCulture) } };
                    var reqReward = new RestRequest(String.Format("/api/v1/nni-aether/update-metrics/{0}/{1}", this.expId, this.trialId), Method.POST, DataFormat.Json);
                    reqReward.AddJsonBody(rewardBody);
                    var resReward = client.Execute(reqReward);
                }
            }

        }

        public async Task OnCancelAsync() {

            Stream stdinStream = Console.OpenStandardInput();
            byte[] result = new byte[10];
            await stdinStream.ReadAsync(result, 0, 1);

            if (this.experimentId != null)
            {
                await this.environment.GetExperiment(this.experimentId).CancelAsync();
                Console.WriteLine(String.Format("Aether Job {0} Cancelled", this.experimentId));
            }
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length < 3)
            {
                Console.WriteLine("usage: AetherClient.exe $restURL $expId $trialId");
                return;
            }
            string restURL = args[0];
            string expId = args[1];
            string trialId = args[2];

            ClientRunner runner = new ClientRunner(restURL, expId, trialId);

            Task[] tasks = new Task[2];
            tasks[0] = runner.runAsync();
            tasks[1] = runner.OnCancelAsync();
            Task.WaitAny(tasks);
        }
    }
}
