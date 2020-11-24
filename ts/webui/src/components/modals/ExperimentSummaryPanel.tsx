import * as React from 'react';
import { downFile } from '../../static/function';
import { Stack, PrimaryButton, DefaultButton, Panel, StackItem } from '@fluentui/react';
import { DRAWEROPTION } from '../../static/const';
import { EXPERIMENT, TRIALS } from '../../static/datamodel';
import { caclMonacoEditorHeight } from '../../static/function';
import MonacoEditor from 'react-monaco-editor';
import '../../static/style/logDrawer.scss';

interface ExpDrawerProps {
    closeExpDrawer: () => void;
    experimentProfile: object;
}

interface ExpDrawerState {
    experiment: string;
    expDrawerHeight: number;
}

class ExperimentSummaryPanel extends React.Component<ExpDrawerProps, ExpDrawerState> {
    public _isExperimentMount!: boolean;
    private refreshId!: number | undefined;

    constructor(props: ExpDrawerProps) {
        super(props);

        this.state = {
            experiment: '',
            expDrawerHeight: window.innerHeight
        };
    }

    getExperimentContent = (): void => {
        const experimentData = JSON.parse(JSON.stringify(this.props.experimentProfile));
        if (experimentData.params.searchSpace) {
            experimentData.params.searchSpace = JSON.parse(experimentData.params.searchSpace);
        }
        const trialMessagesArr = TRIALS.getTrialJobList();
        const interResultList = TRIALS.getMetricsList();
        Object.keys(trialMessagesArr).map(item => {
            // not deal with trial's hyperParameters
            const trialId = trialMessagesArr[item].jobId;
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
            experimentParameters: experimentData,
            trialMessage: trialMessagesArr
        };
        if (this._isExperimentMount === true) {
            this.setState({ experiment: JSON.stringify(result, null, 4) });
        }

        if (['DONE', 'ERROR', 'STOPPED'].includes(EXPERIMENT.status)) {
            if (this.refreshId !== null || this.refreshId !== undefined) {
                window.clearInterval(this.refreshId);
            }
        }
    };

    downExperimentParameters = (): void => {
        const { experiment } = this.state;
        downFile(experiment, 'experiment.json');
    };

    onWindowResize = (): void => {
        this.setState(() => ({ expDrawerHeight: window.innerHeight }));
    };

    componentDidMount(): void {
        this._isExperimentMount = true;
        this.getExperimentContent();
        this.refreshId = window.setInterval(this.getExperimentContent, 10000);
        window.addEventListener('resize', this.onWindowResize);
    }

    componentWillUnmount(): void {
        this._isExperimentMount = false;
        window.clearTimeout(this.refreshId);
        window.removeEventListener('resize', this.onWindowResize);
    }

    render(): React.ReactNode {
        const { closeExpDrawer } = this.props;
        const { experiment, expDrawerHeight } = this.state;
        const monacoEditorHeight = caclMonacoEditorHeight(expDrawerHeight);

        return (
            <Panel isOpen={true} hasCloseButton={false} isLightDismiss={true} onLightDismissClick={closeExpDrawer}>
                <div className='panel'>
                    <div className='panelName'>Experiment summary</div>
                    <MonacoEditor
                        width='100%'
                        height={monacoEditorHeight}
                        language='json'
                        value={experiment}
                        options={DRAWEROPTION}
                    />
                    <Stack horizontal className='buttons'>
                        <StackItem grow={50} className='download'>
                            <PrimaryButton text='Download' onClick={this.downExperimentParameters} />
                        </StackItem>
                        <StackItem grow={50} className='close'>
                            <DefaultButton text='Close' onClick={closeExpDrawer} />
                        </StackItem>
                    </Stack>
                </div>
            </Panel>
        );
    }
}

export default ExperimentSummaryPanel;
