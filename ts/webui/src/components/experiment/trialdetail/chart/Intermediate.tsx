import React, { useEffect, useState } from 'react';
import { Stack, PrimaryButton, Toggle, IStackTokens } from '@fluentui/react';
import { TooltipForIntermediate, AllTrialsIntermediateChart } from '@static/interface';
import { reformatRetiariiParameter } from '@static/function';
import ReactEcharts from 'echarts-for-react';
import 'echarts/lib/component/tooltip';
import 'echarts/lib/component/title';

const stackTokens: IStackTokens = {
    childrenGap: 20
};

interface IntermediateProps {
    source: AllTrialsIntermediateChart[];
}

const Intermediate = (props: IntermediateProps): any => {
    let pointInput!: HTMLInputElement | null;
    let minValInput!: HTMLInputElement | null;
    let maxValInput!: HTMLInputElement | null;

    // const [detailSource, setDetailSource] = useState([] as AllTrialsIntermediateChart[]);
    const [interSource, setInterSource] = useState({} as object);
    const [filterSource, setFilterSource] = useState([] as AllTrialsIntermediateChart[]);
    // const [eachIntermediateNum, setEachIntermediateNum] = useState(1 as number);
    const [isLoadconfirmBtn, setIsLoadconfirmBtn] = useState(false as boolean);
    const [isFilter, setIsFilter] = useState(false as boolean);
    const { source } = props;

    const drawIntermediate = (source: AllTrialsIntermediateChart[]): void => {
        if (source.length > 0) {
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
                                <div class='trial-No'>Trial No.: ${trialNo}</div> 
                                <div class='main'>
                                <div><span>Trial ID: </span>${trialId}</div>
                                <div><span>Intermediate: </span>${data.data}</div>
                                <div><span>Parameters: </span><pre>${JSON.stringify(
                                    reformatRetiariiParameter(parameter),
                                    null,
                                    4
                                )}</pre>
                                </div>
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
                        type: 'inside',
                        yAxisIndex: [0],
                        filterMode: 'none',
                        start: 0, // percent
                        end: 100 // percent
                    }
                ],
                series: source
            };
            setInterSource(option);
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
            setInterSource(nullData);
        }
    };
    // confirm btn function [filter data]
    const filterLines = (): void => {
        const filterSource: AllTrialsIntermediateChart[] = [];
        setIsLoadconfirmBtn(true);
        const pointVal = pointInput !== null ? pointInput.value : '';
        const minVal = minValInput !== null ? minValInput.value : '';
        const maxVal = maxValInput !== null ? maxValInput.value : '';
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
            setFilterSource(filterSource);
            drawIntermediate(filterSource);
        }
        setIsLoadconfirmBtn(false);
    };
    const switchTurn = (ev: React.MouseEvent<HTMLElement>, checked?: boolean): void => {
        setIsFilter(checked ?? true);
    };

    useEffect(() => {
        if (isFilter === true) {
            const pointVal = pointInput !== undefined ? pointInput!.value : '';
            const minVal = minValInput !== undefined ? minValInput!.value : '';
            if (pointVal === '' || minVal === '') {
                drawIntermediate(source);
            } else {
                drawIntermediate(filterSource);
            }
        } else {
            drawIntermediate(source);
        }
    }, [isFilter, source]);

    const intermediateChart = React.useMemo(() => {
        return (
            <div className='intermediate-graph graph'>
                <ReactEcharts
                    option={interSource}
                    style={{ width: '100%', height: 400, margin: '0 auto' }}
                    notMerge={true} // update now
                />
                <div className='fontColor333 xAxis'># Intermediate result</div>
            </div>
        );
    }, [interSource]);
    return (
        <div>
            {/* style in para.scss */}
            <Stack horizontal horizontalAlign='end' tokens={stackTokens} className='meline intermediate'>
                {isFilter ? (
                    <div>
                        <span className='filter-x'># Intermediate result</span>
                        <input
                            // placeholder="point"
                            ref={(input): any => (pointInput = input)}
                            className='strange'
                        />
                        <span>Metric range</span>
                        <input
                            // placeholder="range"
                            ref={(input): any => (minValInput = input)}
                        />
                        <span className='hyphen'>-</span>
                        <input
                            // placeholder="range"
                            ref={(input): any => (maxValInput = input)}
                        />
                        <PrimaryButton
                            className='intermeidateconfirm'
                            text='Confirm'
                            onClick={filterLines}
                            disabled={isLoadconfirmBtn}
                        />
                    </div>
                ) : null}
                {/* filter message */}
                <Toggle label='Filter' inlineLabel onChange={switchTurn} />
            </Stack>
            {intermediateChart}
        </div>
    );
};

export default Intermediate;
