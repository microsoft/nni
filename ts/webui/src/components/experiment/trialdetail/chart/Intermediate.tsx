import * as React from 'react';
import { Stack, PrimaryButton, Toggle, IStackTokens } from '@fluentui/react';
import { TooltipForIntermediate, EventMap, AllTrialsIntermediateChart } from '@static/interface';
import { reformatRetiariiParameter } from '@static/function';
import ReactEcharts from 'echarts-for-react';
import 'echarts/lib/component/tooltip';
import 'echarts/lib/component/title';

const stackTokens: IStackTokens = {
    childrenGap: 20
};

interface IntermediateState {
    detailSource: AllTrialsIntermediateChart[];
    interSource: object;
    filterSource: AllTrialsIntermediateChart[];
    eachIntermediateNum: number; // trial's intermediate number count
    isLoadconfirmBtn: boolean;
    isFilter?: boolean | undefined;
    length: number;
    startMediaY: number;
    endMediaY: number;
}

interface IntermediateProps {
    source: AllTrialsIntermediateChart[];
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
            startMediaY: 0,
            endMediaY: 100
        };
    }

    drawIntermediate = (source: AllTrialsIntermediateChart[]): void => {
        if (source.length > 0) {
            this.setState({
                length: source.length,
                detailSource: source
            });
            const { startMediaY, endMediaY } = this.state;
            const xAxis: number[] = [];
            // find having most intermediate number
            source.sort((a, b) => {
                return b.data.length - a.data.length;
            });
            for (let i = 1; i <= source[0].data.length; i++) {
                xAxis.push(i);
            }
            const option = {
                tooltip: {
                    trigger: 'item',
                    enterable: true,
                    confine: true,
                    formatter: function (data: TooltipForIntermediate): React.ReactNode {
                        const trialId = data.seriesName;
                        // parameter and trialNo need to have the init value otherwise maybe cause page broke down
                        let parameter = {};
                        let trialNo = 0;
                        const renderTrial = source.find(key => key.name === trialId);
                        if (renderTrial !== undefined) {
                            parameter = renderTrial.parameter;
                            trialNo = renderTrial.sequenceId;
                        }
                        return `
                            <div class="tooldetailAccuracy">
                                <div>Trial No.: ${trialNo}</div> 
                                <div>Trial ID: ${trialId}</div>
                                <div>Intermediate: ${data.data}</div>
                                <div>Parameters: <pre>${JSON.stringify(
                                    reformatRetiariiParameter(parameter),
                                    null,
                                    4
                                )}</pre>
                                </div>
                            </div>
                        `;
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
                series: source
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
        const filterSource: AllTrialsIntermediateChart[] = [];
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
                const position = JSON.parse(pointVal);
                const min = JSON.parse(minVal);
                if (maxVal === '') {
                    // user not input max value
                    for (const item of source) {
                        const val = item.data[position - 1];
                        if (val >= min) {
                            filterSource.push(item);
                        }
                    }
                } else {
                    const max = JSON.parse(maxVal);
                    for (const item of source) {
                        const val = item.data[position - 1];
                        if (val >= min && val <= max) {
                            filterSource.push(item);
                        }
                    }
                }
                this.setState({ filterSource: filterSource });
                this.drawIntermediate(filterSource);
            }
            this.setState({ isLoadconfirmBtn: false });
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
            const { source } = this.props;

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
                            <PrimaryButton
                                className='intermeidateconfirm'
                                text='Confirm'
                                onClick={this.filterLines}
                                disabled={isLoadconfirmBtn}
                            />
                        </div>
                    ) : null}
                    {/* filter message */}
                    <Toggle label='Filter' inlineLabel onChange={this.switchTurn} />
                </Stack>
                <div className='intermediate-graph graph'>
                    <ReactEcharts
                        option={interSource}
                        style={{ width: '100%', height: 400, margin: '0 auto' }}
                        notMerge={true} // update now
                        onEvents={IntermediateEvents}
                    />
                    <div className='fontColor333 xAxis'># Intermediate result</div>
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
