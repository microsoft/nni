import React, { useState, useEffect } from 'react';
import { renderToString } from 'react-dom/server';
import { Stack, Modal, IconButton, IDragOptions, ContextualMenu, Dropdown, IDropdownOption } from '@fluentui/react';
import ReactEcharts from 'echarts-for-react';
import { EXPERIMENT, TRIALS } from '@static/datamodel';
import { Trial } from '@model/trial';
import { TooltipForIntermediate, SingleAxis } from '@static/interface';
import { contentStyles, iconButtonStyles } from '@components/fluent/ModalTheme';
import { convertDuration, parseMetrics, getIntermediateAllKeys } from '@static/function';
import '@style/experiment/trialdetail/compare.scss';

/***
 * Compare file is designed for [each trial intermediate result, trials compare model]
 * if trial has dict intermediate result,
 * graph only supports to show all keys that type is number
 */

function _getWebUIWidth(): number {
    return window.innerWidth;
}

const dragOptions: IDragOptions = {
    moveMenuItemText: 'Move',
    closeMenuItemText: 'Close',
    menu: ContextualMenu
};

// TODO: this should be refactored to the common modules
// copied from trial.ts
function _parseIntermediates(trial: Trial, key: string): number[] {
    const intermediates: number[] = [];
    for (const metric of trial.intermediates) {
        if (metric === undefined) {
            break;
        }
        const parsedMetric = parseMetrics(metric.data);
        if (typeof parsedMetric === 'object') {
            // TODO: should handle more types of metric keys
            intermediates.push(parsedMetric[key]);
        } else {
            intermediates.push(parsedMetric);
        }
    }
    return intermediates;
}

interface Item {
    id: string;
    sequenceId: number;
    duration: string;
    parameters: Map<string, any>;
    metrics: Map<string, any>;
    intermediates: number[];
}

interface CompareProps {
    trials: Trial[];
    title: string;
    onHideDialog: () => void;
    changeSelectTrialIds?: () => void;
}

function CompareIndex(props: CompareProps): any {
    const { trials, title } = props;
    const atrial = trials.find(item => item.intermediates.length > 0);
    const intermediateAllKeysList = getIntermediateAllKeys(atrial === undefined ? trials[0] : atrial);
    const [intermediateKey, setIntermediateKey] = useState(
        intermediateAllKeysList.length > 0 ? intermediateAllKeysList[0] : 'default'
    );
    const runningTrial = trials.find(item => item.status === 'RUNNING');
    const runningTrialIntermediateListLength = runningTrial !== undefined ? runningTrial.intermediates.length : -1;

    function itemsList(): Item[] {
        const inferredSearchSpace = TRIALS.inferredSearchSpace(EXPERIMENT.searchSpaceNew);
        const flatten = (m: Map<SingleAxis, any>): Map<string, any> => {
            return new Map(Array.from(m).map(([key, value]) => [key.baseName, value]));
        };
        return trials.map(trial => ({
            id: trial.id,
            sequenceId: trial.sequenceId,
            duration: convertDuration(trial.duration),
            parameters: flatten(trial.parameters(inferredSearchSpace)),
            metrics: flatten(trial.metrics(TRIALS.inferredMetricSpace())),
            intermediates: _parseIntermediates(trial, intermediateKey)
        }));
    }

    const [items, setItems] = useState(itemsList());

    // react componentDidMount & componentDidUpdate
    useEffect(() => {
        setItems(itemsList());
    }, [intermediateKey, runningTrialIntermediateListLength]); // update condition

    // page related function
    const _generateTooltipSummary = (row: Item, value: string): string =>
        renderToString(
            <div className='tooldetailAccuracy'>
                <div className='main'>
                    <div>Trial No.: {row.sequenceId}</div>
                    <div>Trial ID: {row.id}</div>
                    <div>Intermediate metric: {value}</div>
                </div>
            </div>
        );

    function _intermediates(items: Item[]): React.ReactNode {
        // Precondition: make sure `items` is not empty
        const xAxisMax = Math.max(...items.map(item => item.intermediates.length));
        const xAxis = Array(xAxisMax)
            .fill(0)
            .map((_, i) => i + 1); // [1, 2, 3, ..., xAxisMax]
        const dataForEchart = items.map(item => ({
            name: item.id,
            data: item.intermediates,
            type: 'line'
        }));
        const legend = dataForEchart.map(item => item.name);
        const option = {
            tooltip: {
                trigger: 'item',
                enterable: true,
                confine: true,
                formatter: (data: TooltipForIntermediate): string => {
                    const item = items.find(k => k.id === data.seriesName) as Item;
                    return _generateTooltipSummary(item, data.data);
                }
            },
            grid: {
                left: '5%',
                top: 40,
                containLabel: true
            },
            legend: {
                type: 'scroll',
                right: 40,
                left: legend.length > 6 ? '15%' : null,
                data: legend
            },
            xAxis: {
                type: 'category',
                boundaryGap: false,
                data: xAxis
            },
            yAxis: {
                type: 'value',
                name: 'Metric',
                scale: true
            },
            series: dataForEchart
        };
        return (
            <div className='graph'>
                <ReactEcharts
                    option={option}
                    style={{ width: '100%', height: 418, margin: '0 auto' }}
                    notMerge={true} // update now
                />
            </div>
        );
    }

    function _renderRow(
        key: string,
        rowName: string,
        className: string,
        items: Item[],
        formatter: (item: Item) => string
    ): React.ReactNode {
        return (
            <tr key={key}>
                <td className='column'>{rowName}</td>
                {items.map(item => (
                    <td className={className} key={item.id}>
                        {formatter(item) || '--'}
                    </td>
                ))}
            </tr>
        );
    }

    function _overlapKeys(s: Map<string, any>[]): string[] {
        // Calculate the overlapped keys for multiple
        const intersection: string[] = [];
        for (const i of s[0].keys()) {
            let inAll = true;
            for (const t of s) {
                if (!Array.from(t.keys()).includes(i)) {
                    inAll = false;
                    break;
                }
            }
            if (inAll) {
                intersection.push(i);
            }
        }
        return intersection;
    }

    // render table column ---
    function _columns(items: Item[]): React.ReactNode {
        // Precondition: make sure `items` is not empty
        const width = _getWebUIWidth();
        let scrollClass: string = '';
        if (width > 1200) {
            scrollClass = items.length > 3 ? 'flex' : '';
        } else if (width < 700) {
            scrollClass = items.length > 1 ? 'flex' : '';
        } else {
            scrollClass = items.length > 2 ? 'flex' : '';
        }
        const parameterKeys = _overlapKeys(items.map(item => item.parameters));
        const metricKeys = _overlapKeys(items.map(item => item.metrics));

        return (
            <table className={`compare-modal-table ${scrollClass} fontColor333`}>
                <tbody>
                    {_renderRow('id', 'ID', 'value idList', items, item => item.id)}
                    {_renderRow('trialnum', 'Trial No.', 'value', items, item => item.sequenceId.toString())}
                    {_renderRow('duration', 'Duration', 'value', items, item => item.duration)}
                    {parameterKeys.map(k =>
                        _renderRow(`space_${k}`, k, 'value', items, item => item.parameters.get(k))
                    )}
                    {metricKeys !== undefined
                        ? metricKeys.map(k =>
                              _renderRow(`metrics_${k}`, `Metric: ${k}`, 'value', items, item => item.metrics.get(k))
                          )
                        : null}
                </tbody>
            </table>
        );
    }

    const closeCompareModal = (): void => {
        const { title, changeSelectTrialIds, onHideDialog } = props;
        if (title === 'Compare trials') {
            // eslint-disable-next-line  @typescript-eslint/no-non-null-assertion
            changeSelectTrialIds!();
        }
        onHideDialog();
    };

    const selectOtherKeys = (_event: React.FormEvent<HTMLDivElement>, item?: IDropdownOption): void => {
        if (item !== undefined) {
            setIntermediateKey(item.text);
        }
    };

    return (
        <Modal
            isOpen={true}
            containerClassName={contentStyles.container}
            className='compare-modal'
            allowTouchBodyScroll={true}
            dragOptions={dragOptions}
            onDismiss={closeCompareModal}
        >
            <div>
                <div className={contentStyles.header}>
                    <span>{title}</span>
                    <IconButton
                        styles={iconButtonStyles}
                        iconProps={{ iconName: 'Cancel' }}
                        ariaLabel='Close popup modal'
                        onClick={closeCompareModal}
                    />
                </div>
                {intermediateAllKeysList.length > 1 ||
                (intermediateAllKeysList.length === 1 && intermediateAllKeysList[0] !== 'default') ? (
                    <Stack horizontalAlign='end' className='selectKeys'>
                        <Dropdown
                            className='select'
                            selectedKey={intermediateKey}
                            options={intermediateAllKeysList.map((key, item) => ({
                                key: key,
                                text: intermediateAllKeysList[item]
                            }))}
                            onChange={selectOtherKeys}
                        />
                    </Stack>
                ) : null}
                <Stack className='compare-modal-intermediate'>
                    {_intermediates(items)}
                    <Stack className='compare-yAxis fontColor333'># Intermediate result</Stack>
                </Stack>
                {title === 'Compare trials' && <Stack>{_columns(items)}</Stack>}
            </div>
        </Modal>
    );
}

export default CompareIndex;
