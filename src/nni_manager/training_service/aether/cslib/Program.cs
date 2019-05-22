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

    class Metrics 
    {
        public int parameter_id { get; set; }
        public string trial_job_id { get; set; }
        public string type { get; set; }
        public int sequence { get; set; }
        public float value  { get; set; }
    }

    class RewardBody
    {
        public string jobId {get; set;}
        public string[] metrics { get; set; }
    }

    class ClientRunner
    {
        private AetherEnvironment environment;
        private RestClient client;
        private string expId;   // NNI experiment ID
        private string trialId;
        private ExperimentStatusCode status;
        private string experimentId = null; // Aether experiment ID
        private NNITrialInfo info = null;

        private NNIHparam hparams = null;
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
            try
            {
                var request = new RestRequest(String.Format("/api/v1/nni-aether/trial-meta/{0}/{1}", this.expId, this.trialId), DataFormat.Json);
                var response = this.client.Get(request);

                var serializer = new RestSharp.Serialization.Json.JsonSerializer();
                info = serializer.Deserialize<NNITrialInfo>(response);

                Console.WriteLine(String.Format("Loading graph from file: {0}", info.aetherConfig.baseGraph));
                IVisualGraph graph = environment.DeserializeGraph(File.ReadAllText(info.aetherConfig.baseGraph));

                Console.WriteLine(String.Format("Setting hyper-parameters: {0}", info.form.hyperParameters.value));
                hparams = JsonConvert.DeserializeObject<NNIHparam>(info.form.hyperParameters.value);
                foreach (var item in hparams.parameters)
                {
                    try {
                        graph.Parameters.First(e => e.Name == item.Key).Value = item.Value;
                    } catch (NullReferenceException e) {
                        Console.WriteLine(String.Format("WARNING: hyper-parameter {0} found in Current Aether Experiment", item.Key));
                    }
                }

                this.experimentId = environment.SubmitExperiment(new ExperimentCreationInfo { Description = String.Format("NNI Experiment {0}/{1}", this.expId, this.trialId) }, graph);
                Console.WriteLine(String.Format("Experiment Submitted: {0}", this.experimentId));

                GuidBody body = new GuidBody { guid = this.experimentId };

                var reqGuid = new RestRequest(String.Format("/api/v1/nni-aether/update-guid/{0}/{1}", this.expId, this.trialId), Method.POST, DataFormat.Json);
                reqGuid.AddJsonBody(body);
                var resGuid = this.client.Execute(reqGuid);
                Console.WriteLine(resGuid.Content);
            }
            catch (Exception e) {
                Console.Error.WriteLine(String.Format("Exception while submitting trial job: {0}", e));
                Console.Error.Flush();
                System.Environment.Exit(1);
            }
            #endregion


            #region updateTrialStatus
            while (true)
            {
                try
                {
                    ExperimentStatusCode statusTmp = environment.GetExperiment(this.experimentId).GetStatus();
                    Console.WriteLine(String.Format("Experiment {0} status updated: {1}", this.experimentId, statusTmp));
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
                catch (Exception e)
                {
                    Console.Error.WriteLine(String.Format("Exception while updating trial status: {0}", e));
                    Console.Error.Flush();
                    System.Environment.Exit(1);
                }
            }


            #endregion

            if (this.status == ExperimentStatusCode.Finished)
            {
                try
                {
                    Console.WriteLine(String.Format("Aether Job {0} finished", this.experimentId));
                    INodeExecution nodeExec = environment.GetExperiment(this.experimentId).GetExecutionGraph().GetNodeExecution(info.aetherConfig.outputNodeAlias);

                    string outputfile = String.Format("{0}\\aether_output.txt", info.workingDirectory);
                    nodeExec.GetOutput(info.aetherConfig.outputName).Download(outputfile);
                    using (StreamReader reader = new StreamReader(outputfile))
                    {
                        string line = reader.ReadToEnd();
                        Metrics metrics = new Metrics {
                            parameter_id = hparams.parameter_id,
                            trial_job_id = this.trialId,
                            type = "FINAL",
                            sequence = 0,
                            value = float.Parse(line, CultureInfo.InvariantCulture)
                        };
                        RewardBody rewardBody = new RewardBody { 
                            jobId = this.trialId, 
                            metrics = new string[1]{
                                JsonConvert.SerializeObject(metrics),
                            }
                        };
                        var reqReward = new RestRequest(String.Format("/api/v1/nni-aether/update-metrics/{0}/{1}", this.expId, this.trialId), Method.POST, DataFormat.Json);
                        reqReward.AddJsonBody(rewardBody);
                        var resReward = client.Execute(reqReward);
                    }
                }
                catch (Exception e)
                {
                    Console.Error.WriteLine(String.Format("Exception while collecting final metric: {0}", e));
                    Console.Error.Flush();
                    System.Environment.Exit(1);
                }
            }

        }

        public async Task OnCancelAsync() {
            try
            {
                Stream stdinStream = Console.OpenStandardInput();
                byte[] result = new byte[10];
                await stdinStream.ReadAsync(result, 0, 1);

                if (this.experimentId != null)
                {
                    await this.environment.GetExperiment(this.experimentId).CancelAsync();
                    Console.WriteLine(String.Format("Aether Job {0} Cancelled", this.experimentId));
                }
            }
            catch (Exception e) {
                Console.Error.WriteLine(String.Format("Exception while cancelling job: {0}", e));
                Console.Error.Flush();
                System.Environment.Exit(1);
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
