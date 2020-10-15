import * as React from 'react';
import { Stack, PrimaryButton, Toggle, IStackTokens } from '@fluentui/react';
import { TooltipForIntermediate, TableObj, Intermedia, EventMap } from '../../static/interface';
import ReactEcharts from 'echarts-for-react';
import 'echarts/lib/component/tooltip';
import 'echarts/lib/component/title';

const stackTokens: IStackTokens = {
    childrenGap: 20
};

interface IntermediateState {
    detailSource: Array<TableObj>;
    interSource: object;
    filterSource: Array<TableObj>;
    eachIntermediateNum: number; // trial's intermediate number count
    isLoadconfirmBtn: boolean;
    isFilter?: boolean | undefined;
    length: number;
    clickCounts: number; // user filter intermediate click confirm btn's counts
    startMediaY: number;
    endMediaY: number;
}

interface IntermediateProps {
    source: Array<TableObj>;
    whichChart: string;
}

class Intermediate extends React.Component<IntermediateProps, IntermediateState> {
    static intervalMediate = 1;
    public pointInput!: HTMLInputElement | null;
    public minValInput!: HTMLInputElement | null;
    public maxValInput!: HTMLInputElement | null;

    constructor(props: IntermediateProps) {
        super(props);
        this.state = {
            detailSource: [],
            interSource: {},
            filterSource: [],
            eachIntermediateNum: 1,
            isLoadconfirmBtn: false,
            isFilter: false,
            length: 100000,
            clickCounts: 0,
            startMediaY: 0,
            endMediaY: 100
        };
    }

    drawIntermediate = (source: Array<TableObj>): void => {
        if (source.length > 0) {
            this.setState({
                length: source.length,
                detailSource: source
            });
            const { startMediaY, endMediaY } = this.state;
            const trialIntermediate: Array<Intermedia> = [];
            Object.keys(source).map(item => {
                const temp = source[item];
                trialIntermediate.push({
                    name: temp.id,
                    data: temp.description.intermediate,
                    type: 'line',
                    hyperPara: temp.description.parameters
                });
            });
            // find max intermediate number
            trialIntermediate.sort((a, b) => {
                return b.data.length - a.data.length;
            });
            const legend: string[] = [];
            // max length
            const length = trialIntermediate[0].data.length;
            const xAxis: number[] = [];
            Object.keys(trialIntermediate).map(item => {
                const temp = trialIntermediate[item];
                legend.push(temp.name);
            });
            for (let i = 1; i <= length; i++) {
                xAxis.push(i);
            }
            const option = {
                tooltip: {
                    trigger: 'item',
                    enterable: true,
                    position: function(point: number[], data: TooltipForIntermediate): number[] {
                        if (data.dataIndex < length / 2) {
                            return [point[0], 80];
                        } else {
                            return [point[0] - 300, 80];
                        }
                    },
                    formatter: function(data: TooltipForIntermediate): React.ReactNode {
                        const trialId = data.seriesName;
                        let obj = {};
                        const temp = trialIntermediate.find(key => key.name === trialId);
                        if (temp !== undefined) {
                            obj = temp.hyperPara;
                        }
                        return (
                            '<div class="tooldetailAccuracy">' +
                            '<div>Trial ID: ' +
                            trialId +
                            '</div>' +
                            '<div>Intermediate: ' +
                            data.data +
                            '</div>' +
                            '<div>Parameters: ' +
                            '<pre>' +
                            JSON.stringify(obj, null, 4) +
                            '</pre>' +
                            '</div>' +
                            '</div>'
                        );
                    }
                },
                grid: {
                    left: '5%',
                    top: 40,
                    containLabel: true
                },
                xAxis: {
                    type: 'category',
                    // name: '# Intermediate',
                    boundaryGap: false,
                    data: xAxis
                },
                yAxis: {
                    type: 'value',
                    name: 'Metric',
                    scale: true
                },
                dataZoom: [
                    {
                        id: 'dataZoomY',
                        type: 'inside',
                        yAxisIndex: [0],
                        filterMode: 'none',
                        start: startMediaY,
                        end: endMediaY
                    }
                ],
                series: trialIntermediate
            };
            this.setState({
                interSource: option
            });
        } else {
            const nullData = {
                grid: {
                    left: '5%',
                    top: 40,
                    containLabel: true
                },
                xAxis: {
                    type: 'category',
                    boundaryGap: false
                },
                yAxis: {
                    type: 'value',
                    name: 'Metric'
                }
            };
            this.setState({ interSource: nullData });
        }
    };

    // confirm btn function [filter data]
    filterLines = (): void => {
        const filterSource: Array<TableObj> = [];
        this.setState({ isLoadconfirmBtn: true }, () => {
            const { source } = this.props;
            // get input value
            const pointVal = this.pointInput !== null ? this.pointInput.value : '';
            const minVal = this.minValInput !== null ? this.minValInput.value : '';
            const maxVal = this.maxValInput !== null ? this.maxValInput.value : '';
            // user not input message
            if (pointVal === '' || minVal === '') {
                alert('Please input filter message');
            } else {
                // user not input max value
                const position = JSON.parse(pointVal);
                const min = JSON.parse(minVal);
                if (maxVal === '') {
                    Object.keys(source).map(item => {
                        const temp = source[item];
                        const val = temp.description.intermediate[position - 1];
                        if (val >= min) {
                            filterSource.push(temp);
                        }
                    });
                } else {
                    const max = JSON.parse(maxVal);
                    Object.keys(source).map(item => {
                        const temp = source[item];
                        const val = temp.description.intermediate[position - 1];
                        if (val >= min && val <= max) {
                            filterSource.push(temp);
                        }
                    });
                }
                this.setState({ filterSource: filterSource });
                this.drawIntermediate(filterSource);
            }
            const counts = this.state.clickCounts + 1;
            this.setState({ isLoadconfirmBtn: false, clickCounts: counts });
        });
    };

    switchTurn = (ev: React.MouseEvent<HTMLElement>, checked?: boolean): void => {
        this.setState({ isFilter: checked });
        if (checked === false) {
            this.drawIntermediate(this.props.source);
        }
    };

    componentDidMount(): void {
        const { source } = this.props;
        this.drawIntermediate(source);
    }

    componentDidUpdate(prevProps: IntermediateProps, prevState: any): void {
        if (this.props.source !== prevProps.source || this.state.isFilter !== prevState.isFilter) {
            const { isFilter, filterSource } = this.state;
            const { whichChart, source } = this.props;

            if (whichChart === 'Intermediate result') {
                if (isFilter === true) {
                    const pointVal = this.pointInput !== null ? this.pointInput.value : '';
                    const minVal = this.minValInput !== null ? this.minValInput.value : '';
                    if (pointVal === '' && minVal === '') {
                        this.drawIntermediate(source);
                    } else {
                        this.drawIntermediate(filterSource);
                    }
                } else {
                    this.drawIntermediate(source);
                }
            }
        }
    }

    render(): React.ReactNode {
        const { interSource, isLoadconfirmBtn, isFilter } = this.state;
        const IntermediateEvents = { dataZoom: this.intermediateDataZoom };

        return (
            <div>
                {/* style in para.scss */}
                <Stack horizontal horizontalAlign='end' tokens={stackTokens} className='meline intermediate'>
                    {isFilter ? (
                        <div>
                            <span className='filter-x'># Intermediate result</span>
                            <input
                                // placeholder="point"
                                ref={(input): any => (this.pointInput = input)}
                                className='strange'
                            />
                            <span>Metric range</span>
                            <input
                                // placeholder="range"
                                ref={(input): any => (this.minValInput = input)}
                            />
                            <span className='hyphen'>-</span>
                            <input
                                // placeholder="range"
                                ref={(input): any => (this.maxValInput = input)}
                            />
                            <PrimaryButton text='Confirm' onClick={this.filterLines} disabled={isLoadconfirmBtn} />
                        </div>
                    ) : null}
                    {/* filter message */}
                    <Stack horizontal className='filter-toggle'>
                        <span>Filter</span>
                        <Toggle onChange={this.switchTurn} />
                    </Stack>
                </Stack>
                <div className='intermediate-graph'>
                    <ReactEcharts
                        option={interSource}
                        style={{ width: '100%', height: 400, margin: '0 auto' }}
                        notMerge={true} // update now
                        onEvents={IntermediateEvents}
                    />
                    <div className='xAxis'># Intermediate result</div>
                </div>
            </div>
        );
    }

    private intermediateDataZoom = (e: EventMap): void => {
        if (e.batch !== undefined) {
            this.setState(() => ({
                startMediaY: e.batch[0].start !== null ? e.batch[0].start : 0,
                endMediaY: e.batch[0].end !== null ? e.batch[0].end : 100
            }));
        }
    };
}

export default Intermediate;
