import * as React from 'react';
import { Stack, IStackTokens, Dropdown } from 'office-ui-fabric-react';
import { EXPERIMENT, TRIALS } from '../static/datamodel';
import { Trial } from '../static/model/trial';
import Title1 from './overview/Title1';
import SuccessTable from './overview/SuccessTable';
import Progressed from './overview/Progress';
import Accuracy from './overview/Accuracy';
import SearchSpace from './overview/SearchSpace';
import BasicInfo from './overview/BasicInfo';
import TrialInfo from './overview/TrialProfile';
import '../static/style/overview.scss';
import '../static/style/logPath.scss';

interface OverviewProps {
    experimentUpdateBroadcast: number;
    trialsUpdateBroadcast: number;
    metricGraphMode: 'max' | 'min';
    bestTrialEntries: string;
    changeMetricGraphMode: (val: 'max' | 'min') => void;
    changeEntries: (entries: string) => void;
}

interface OverviewState {
    trialConcurrency: number;
}

class Overview extends React.Component<OverviewProps, OverviewState> {
    constructor(props: OverviewProps) {
        super(props);
        this.state = {
            trialConcurrency: EXPERIMENT.trialConcurrency
        };
    }

    clickMaxTop = (event: React.SyntheticEvent<EventTarget>): void => {
        event.stopPropagation();
        // #999 panel active bgcolor; #b3b3b3 as usual
        const { changeMetricGraphMode } = this.props;
        changeMetricGraphMode('max');
    }


    clickMinTop = (event: React.SyntheticEvent<EventTarget>): void => {
        event.stopPropagation();
        const { changeMetricGraphMode } = this.props;
        changeMetricGraphMode('min');
    }

    changeConcurrency = (val: number): void => {
        this.setState({ trialConcurrency: val });
    }

    // updateEntries = (event: React.FormEvent<HTMLDivElement>, item: IDropdownOption | undefined): void => {
    updateEntries = (event: React.FormEvent<HTMLDivElement>, item: any): void => {
        if (item !== undefined) {
            this.props.changeEntries(item.key);
        }
    }

    render(): React.ReactNode {
        const { trialConcurrency } = this.state;
        const { experimentUpdateBroadcast, metricGraphMode, bestTrialEntries } = this.props;
        const bestTrials = this.findBestTrials();
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        const bestAccuracy = bestTrials.length > 0 ? bestTrials[0].accuracy! : NaN;
        const accuracyGraphData = this.generateAccuracyGraph(bestTrials);
        const noDataMessage = bestTrials.length > 0 ? '' : 'No data';

        const titleMaxbgcolor = (metricGraphMode === 'max' ? '#333' : '#b3b3b3');
        const titleMinbgcolor = (metricGraphMode === 'min' ? '#333' : '#b3b3b3');

        const stackTokens: IStackTokens = {
            childrenGap: 30,
        };

        const entriesOption = [
            { key: '10', text: 'Display top 10 trials' },
            { key: '20', text: 'Display top 20 trials' },
            { key: '30', text: 'Display top 30 trials' },
            { key: '50', text: 'Display top 50 trials' },
            { key: '100', text: 'Display top 100 trials' }
        ];
        return (
            <div className="overview">
                {/* status and experiment block */}
                <Stack className="bottomDiv bgNNI">
                    <Title1 text="Experiment" icon="11.png" />
                    <BasicInfo experimentUpdateBroadcast={experimentUpdateBroadcast} />
                </Stack>

                <Stack horizontal className="overMessage bottomDiv">
                    {/* status block */}
                    <Stack.Item grow className="prograph bgNNI borderRight">
                        <Title1 text="Status" icon="5.png" />
                        <Progressed
                            bestAccuracy={bestAccuracy}
                            concurrency={trialConcurrency}
                            changeConcurrency={this.changeConcurrency}
                            experimentUpdateBroadcast={experimentUpdateBroadcast}
                        />
                    </Stack.Item>
                    {/* experiment parameters search space tuner assessor... */}
                    <Stack.Item grow styles={{root: {width: 450}}} className="overviewBoder borderRight bgNNI">
                        <Title1 text="Search space" icon="10.png" />
                        <Stack className="experiment">
                            <SearchSpace searchSpace={EXPERIMENT.searchSpace} />
                        </Stack>
                    </Stack.Item>
                    {/* <Stack.Item grow styles={{root: {width: 450}}} className="bgNNI"> */}
                    <Stack.Item grow styles={{root: {width: 450}}} className="bgNNI">
                        <Title1 text="Config" icon="4.png" />
                        <Stack className="experiment">
                            {/* the scroll bar all the trial profile in the searchSpace div*/}
                            <div className="experiment searchSpace">
                                <TrialInfo
                                    experimentUpdateBroadcast={experimentUpdateBroadcast}
                                    concurrency={trialConcurrency}
                                />
                            </div>
                        </Stack>
                    </Stack.Item>
                </Stack>

                <Stack style={{backgroundColor: '#fff'}}>
                    <Stack horizontal className="top10bg" style={{position: 'relative'}}>
                        <div
                            className="title"
                            onClick={this.clickMaxTop}
                        >
                            <Title1 text="Top maximal trials" icon="max.png" fontColor={titleMaxbgcolor} />
                        </div>
                        <div
                            className="title minTitle"
                            onClick={this.clickMinTop}
                        >
                            <Title1 text="Top minimal trials" icon="min.png" fontColor={titleMinbgcolor} />
                        </div>
                        <div style={{position: 'absolute', right: 52, top: 6}}>
                            <Dropdown
                                selectedKey={bestTrialEntries}
                                options={entriesOption}
                                onChange={this.updateEntries}
                                styles={{ root: { width: 170 } }}
                            />
                        </div>
                    </Stack>
                    <Stack horizontal tokens={stackTokens}>
                        <div style={{ width: '40%', position: 'relative'}}>
                            <Accuracy
                                accuracyData={accuracyGraphData}
                                accNodata={noDataMessage}
                                height={404}
                            />
                        </div>
                        <div style={{ width: '60%'}}>
                            <SuccessTable trialIds={bestTrials.map(trial => trial.info.id)} />
                        </div>
                    </Stack>
                </Stack>
            </div>
        );
    }

    private findBestTrials(): Trial[] {
        const bestTrials = TRIALS.sort();
        const { bestTrialEntries } = this.props;
        if (this.props.metricGraphMode === 'max') {
            bestTrials.reverse().splice(JSON.parse(bestTrialEntries));
        } else {
            bestTrials.splice(JSON.parse(bestTrialEntries));
        }
        return bestTrials;
    }

    private generateAccuracyGraph(bestTrials: Trial[]): object {
        const xSequence = bestTrials.map(trial => trial.sequenceId);
        const ySequence = bestTrials.map(trial => trial.accuracy);

        return {
            // support max show 0.0000000
            grid: {
                left: 67,
                right: 40
            },
            tooltip: {
                trigger: 'item'
            },
            xAxis: {
                name: 'Trial',
                type: 'category',
                data: xSequence
            },
            yAxis: {
                name: 'Default metric',
                type: 'value',
                scale: true,
                data: ySequence
            },
            series: [{
                symbolSize: 6,
                type: 'scatter',
                data: ySequence
            }]
        };
    }
}

export default Overview;
