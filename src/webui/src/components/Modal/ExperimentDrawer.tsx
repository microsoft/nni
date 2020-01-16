import * as React from 'react';
import axios from 'axios';
import { downFile } from '../../static/function';
import {
    Stack, PrimaryButton, Panel, StackItem, Pivot, PivotItem
} from 'office-ui-fabric-react';
import { MANAGER_IP, DRAWEROPTION } from '../../static/const';
import MonacoEditor from 'react-monaco-editor';
import '../../static/style/logDrawer.scss';

interface ExpDrawerProps {
    isVisble: boolean;
    closeExpDrawer: () => void;
}

interface ExpDrawerState {
    experiment: string;
    expDrawerHeight: number;
}

class ExperimentDrawer extends React.Component<ExpDrawerProps, ExpDrawerState> {

    public _isCompareMount!: boolean;
    constructor(props: ExpDrawerProps) {
        super(props);

        this.state = {
            experiment: '',
            expDrawerHeight: window.innerHeight
        };
    }

    getExperimentContent = (): void => {
        axios
            .all([
                axios.get(`${MANAGER_IP}/experiment`),
                axios.get(`${MANAGER_IP}/trial-jobs`),
                axios.get(`${MANAGER_IP}/metric-data`)
            ])
            .then(axios.spread((res, res1, res2) => {
                if (res.status === 200 && res1.status === 200 && res2.status === 200) {
                    if (res.data.params.searchSpace) {
                        res.data.params.searchSpace = JSON.parse(res.data.params.searchSpace);
                    }
                    const trialMessagesArr = res1.data;
                    const interResultList = res2.data;
                    Object.keys(trialMessagesArr).map(item => {
                        // not deal with trial's hyperParameters
                        const trialId = trialMessagesArr[item].id;
                        // add intermediate result message
                        trialMessagesArr[item].intermediate = [];
                        Object.keys(interResultList).map(key => {
                            const interId = interResultList[key].trialJobId;
                            if (trialId === interId) {
                                trialMessagesArr[item].intermediate.push(interResultList[key]);
                            }
                        });
                    });
                    const result = {
                        experimentParameters: res.data,
                        trialMessage: trialMessagesArr
                    };
                    if (this._isCompareMount === true) {
                        this.setState({ experiment: JSON.stringify(result, null, 4) });
                    }
                }
            }));
    }

    downExperimentParameters = (): void => {
        const { experiment } = this.state;
        downFile(experiment, 'experiment.json');
    }

    onWindowResize = (): void => {
        this.setState(() => ({ expDrawerHeight: window.innerHeight }));
    }

    componentDidMount(): void {
        this._isCompareMount = true;
        this.getExperimentContent();
        window.addEventListener('resize', this.onWindowResize);
    }

    componentWillReceiveProps(nextProps: ExpDrawerProps): void {
        const { isVisble } = nextProps;
        if (isVisble === true) {
            this.getExperimentContent();
        }
    }

    componentWillUnmount(): void {
        this._isCompareMount = false;
        window.removeEventListener('resize', this.onWindowResize);
    }

    render(): React.ReactNode {
        const { isVisble, closeExpDrawer } = this.props;
        const { experiment, expDrawerHeight } = this.state;
        return (
            <Stack className="logDrawer">
                <Panel
                    isOpen={isVisble}
                    onDismiss={closeExpDrawer}
                    styles={{ root: { height: expDrawerHeight } }}
                >
                    <Pivot style={{ height: expDrawerHeight - 64, minHeight: 190 }} className="log-tab-body">
                        <PivotItem headerText="Experiment Parameters">
                            <div className="just-for-log">
                                <MonacoEditor
                                    width="100%"
                                    height={expDrawerHeight - 144}
                                    language="json"
                                    value={experiment}
                                    options={DRAWEROPTION}
                                />
                            </div>
                            <Stack horizontal className="buttons">
                                <StackItem grow={50} className="download">
                                    <PrimaryButton
                                        text="Download"
                                        onClick={this.downExperimentParameters}
                                    />
                                </StackItem>
                                <StackItem grow={50} className="close">
                                    <PrimaryButton
                                        text="Close"
                                        onClick={closeExpDrawer}
                                    />
                                </StackItem>
                            </Stack>
                        </PivotItem>
                    </Pivot>
                </Panel>
            </Stack>
        );
    }
}

export default ExperimentDrawer;
