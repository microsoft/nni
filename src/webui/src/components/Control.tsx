import * as React from 'react';
import { Input, Button, message } from 'antd';
import axios from 'axios';
import { MANAGER_IP, CONTROLTYPE } from '../static/const';
const { TextArea } = Input;
import '../static/style/control.scss';

interface ExperimentParams {
    authorName: string;
    experimentName: string;
    trialConcurrency: number;
    maxExecDuration: number;
    maxTrialNum: number;
    searchSpace: string;
    tuner: {
        tunerCommand: string;
        tunerCwd: string;
        tunerCheckpointDirectory: string;
        tunerGpuNum?: number;
    };
    assessor?: {
        assessorCommand: string;
        assessorCwd: string;
        assessorCheckpointDirectory: string;
        assessorGpuNum?: number;
    };
}

interface Experiments {
    params: ExperimentParams;
    id: string;
    startTime?: Date;
    endTime?: Date;
    revision: number;
    execDuration: number;
}

interface TrialNumber {
    maxExecDuration: number;
    trialConcurrency: number;
}

interface ControlState {
    addisabled: boolean;
    addTrial: string;
    updateSearch: string;
    trialNum: TrialNumber;
    trialMess: string;
    updisabled: boolean;
    upTrialdis: boolean;
    experiment: Experiments;
}

class Control extends React.Component<{}, ControlState> {

    public _isMounted = false;

    constructor(props: {}) {

        super(props);
        this.state = {
            addisabled: false,
            upTrialdis: false,
            addTrial: '',
            updateSearch: '',
            updisabled: false,
            trialNum: {
                maxExecDuration: 0,
                trialConcurrency: 0
            },
            trialMess: '',
            // experiment origin data obj
            experiment: {
                params: {
                    authorName: '',
                    experimentName: '',
                    trialConcurrency: 0,
                    maxExecDuration: 0,
                    maxTrialNum: 0,
                    searchSpace: '',
                    tuner: {
                        tunerCommand: '',
                        tunerCwd: '',
                        tunerCheckpointDirectory: '',
                    }
                },
                id: '',
                revision: 0,
                execDuration: 0,
            }
        };
    }
    
    updateTrialNumLoad = () => {
        if (this._isMounted) {
            this.setState({
                upTrialdis: true,
            });
        }
    }

    updateTrialNumNormal = () => {
        if (this._isMounted) {
            this.setState({
                upTrialdis: false,
            });
        }
    }

    addButtonLoad = () => {
        if (this._isMounted) {
            this.setState({
                addisabled: true
            });
        }
    }

    addButtonNormal = () => {
        if (this._isMounted) {
            this.setState({
                addisabled: false,
            });
        }
    }

    updateSearchLoad = () => {
        if (this._isMounted) {
            this.setState({
                updisabled: true,
            });
        }
    }

    updateSearchNormal = () => {
        if (this._isMounted) {
            this.setState({
                updisabled: false,
            });
        }
    }

    getTrialNum = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
        if (this._isMounted) {
            this.setState({
                trialMess: event.target.value
            });
        }
    }

    getAddTrialval = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
        if (this._isMounted) {
            this.setState({
                addTrial: event.target.value
            });
        }
    }

    updateSearchCon = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
        if (this._isMounted) {
            this.setState({
                updateSearch: event.target.value
            });
        }
    }

    // get add trial example
    getAddExample = () => {
        axios(`${MANAGER_IP}/trial-jobs`, {
            method: 'GET'
        }).then(res => {
            if (res.status === 200 && this._isMounted) {
                if (res.data.length !== 0) {
                    const addTrialExam = JSON.parse(res.data[0].hyperParameters).parameters;
                    this.setState({
                        addTrial: JSON.stringify(addTrialExam, null, 4)
                    });
                }
            }
        });
    }

    // get update search_space file and experiment
    getUpdateExample = () => {
        axios(`${MANAGER_IP}/experiment`, {
            method: 'GET'
        }).then(res => {
            if (res.status === 200 && this._isMounted) {
                const sespaceExam = JSON.parse(res.data.params.searchSpace);
                const trialnum: Array<TrialNumber> = [];
                trialnum.push({
                    maxExecDuration: res.data.params.maxExecDuration,
                    trialConcurrency: res.data.params.trialConcurrency
                });
                this.setState(() => ({
                    updateSearch: JSON.stringify(sespaceExam, null, 4),
                    trialNum: trialnum[0],
                    trialMess: JSON.stringify(trialnum[0], null, 4),
                    experiment: res.data
                }));
            }
        });
    }

    // update trial number parameters
    trialParameterMess = (exper: Experiments, str: string) => {
    
        axios(`${MANAGER_IP}/experiment`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json;charset=utf-8'
            },
            data: exper,
            params: {
                update_type: str,
            }
        }).then(res => {
            if (res.status === 200) {
                message.success(`Update ${str.toLocaleLowerCase()} successfully`);
                this.getUpdateExample();
            } else {
                message.error(`Update ${str.toLocaleLowerCase()} failed`);
            }
        });
    }

    updateTrialMess = () => {
        const { trialMess } = this.state;
        if (trialMess !== '' || trialMess !== null) {
            this.updateTrialNumLoad();
            const { experiment } = this.state;
            const newExperiment = JSON.parse(JSON.stringify(experiment));
            const trialObj = JSON.parse(trialMess);
            const orimaxDuration = experiment.params.maxExecDuration;
            const oriconTrial = experiment.params.trialConcurrency;
            const flagMax = (trialObj.maxExecDuration !== orimaxDuration);
            const flagCon = (trialObj.trialConcurrency !== oriconTrial);
            if (flagCon && flagMax) {
                newExperiment.params.trialConcurrency = trialObj.trialConcurrency;
                newExperiment.params.maxExecDuration = trialObj.maxExecDuration;
                this.trialParameterMess(newExperiment, CONTROLTYPE[1]);
                this.trialParameterMess(newExperiment, CONTROLTYPE[2]);
            } else if (flagCon) {
                newExperiment.params.trialConcurrency = trialObj.trialConcurrency;
                this.trialParameterMess(newExperiment, CONTROLTYPE[1]);
            } else if (flagMax) {
                newExperiment.params.maxExecDuration = trialObj.maxExecDuration;
                this.trialParameterMess(newExperiment, CONTROLTYPE[2]);
            } else {
                message.info('you have not modified this file');
            }
            this.updateTrialNumNormal();
        } else {
            message.error('The text can not be empty');
        }
    }

    userSubmitJob = () => {
        const { addTrial } = this.state;
        if (addTrial === null || addTrial === '') {
            message.error('The text can not be empty');
        } else {
            this.addButtonLoad();
            // new experiment obj
            axios(`${MANAGER_IP}/trial-jobs`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                data: addTrial
            }).then(res => {
                if (res.status === 200) {
                    message.success('Submit successfully');
                } else {
                    message.error('Submit failed');
                }
                this.addButtonNormal();
            });
        }

    }

    userUpdateSeaspace = () => {

        this.updateSearchLoad();
        const { updateSearch } = this.state;
        if (updateSearch !== '' || updateSearch !== null) {
            const { experiment } = this.state;
            const newExperiment = JSON.parse(JSON.stringify(experiment));
            newExperiment.params.searchSpace = updateSearch;
            this.trialParameterMess(newExperiment, CONTROLTYPE[0]);
            this.updateSearchNormal();
        } else {
            message.error('The text can not be empty');
        }
    }

    componentDidMount() {

        this._isMounted = true;
        this.getAddExample();
        this.getUpdateExample();
    }

    componentWillUnmount() {

        this._isMounted = false;
    }

    render() {
        const { addTrial, addisabled, updateSearch, updisabled,
            trialMess, upTrialdis
        } = this.state;
        return (
            <div className="user">
                <div className="userCon">
                    <div className="addtrial">
                        <div className="addtitle">
                            <span className="line">|</span>
                            Experiment parameters
                        </div>
                        <div className="userInput">
                            <TextArea
                                value={trialMess}
                                autosize={{ minRows: 9 }}
                                onChange={this.getTrialNum}
                            />
                        </div>
                        <div className="addBtubox">
                            <Button
                                className="userSubmit"
                                type="primary"
                                onClick={this.updateTrialMess}
                                disabled={upTrialdis}
                            >
                                Update
                            </Button>
                        </div>
                    </div>
                    <div className="clear" />
                    <div className="addtrial">
                        <div className="addtitle">
                            <span className="line">|</span>
                            Add New Trail
                        </div>
                        <div className="userInput">
                            <TextArea
                                id="userInputJob"
                                value={addTrial}
                                autosize={{ minRows: 9 }}
                                onChange={this.getAddTrialval}
                            />
                        </div>
                        <div className="addBtubox">
                            <Button
                                className="userSubmit"
                                type="primary"
                                onClick={this.userSubmitJob}
                                disabled={addisabled}
                            >
                                Submit
                            </Button>
                        </div>
                    </div>
                    {/* clear float */}
                    <div className="clear" />
                    <div className="searchbox">
                        <div className="updatesear">
                            <span className="line">|</span>
                            user update search_space file
                        </div>
                        <div className="userInput">
                            <TextArea
                                id="InputUpdate"
                                autosize={{ minRows: 20 }}
                                value={updateSearch}
                                onChange={this.updateSearchCon}
                            />
                        </div>
                        <div className="addBtubox">
                            <Button
                                className="buttonbac"
                                type="primary"
                                onClick={this.userUpdateSeaspace}
                                disabled={updisabled}
                            >
                                Update
                            </Button>
                        </div>
                    </div>
                </div>
            </div>
        );
    }
}
export default Control;