import * as React from 'react';
import { renderToString } from 'react-dom/server';
import { Stack, Modal, IconButton, IDragOptions, ContextualMenu } from '@fluentui/react';
import ReactEcharts from 'echarts-for-react';
import { TooltipForIntermediate, TableObj, SingleAxis } from '../../static/interface';
import { contentStyles, iconButtonStyles } from '../buttons/ModalTheme';
import { convertDuration, parseMetrics } from '../../static/function';
import { EXPERIMENT, TRIALS } from '../../static/datamodel';
import '../../static/style/compare.scss';

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
function _parseIntermediates(trial: TableObj): number[] {
    const intermediates: number[] = [];
    for (const metric of trial.intermediates) {
        if (metric === undefined) {
            break;
        }
        const parsedMetric = parseMetrics(metric.data);
        if (typeof parsedMetric === 'object') {
            // TODO: should handle more types of metric keys
            intermediates.push(parsedMetric.default);
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
    trials: TableObj[];
    title: string;
    showDetails: boolean;
    onHideDialog: () => void;
}

class Compare extends React.Component<CompareProps, {}> {
    constructor(props: CompareProps) {
        super(props);
    }

    private _generateTooltipSummary = (row: Item, value: string): string =>
        renderToString(
            <div className='tooldetailAccuracy'>
                <div>Trial No.: {row.sequenceId}</div>
                <div>Trial ID: {row.id}</div>
                <div>Intermediate metric: {value}</div>
            </div>
        );

    private _intermediates(items: Item[]): React.ReactNode {
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
                    return this._generateTooltipSummary(item, data.data);
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

    private _renderRow(
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

    private _overlapKeys(s: Map<string, any>[]): string[] {
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
    private _columns(items: Item[]): React.ReactNode {
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
        const parameterKeys = this._overlapKeys(items.map(item => item.parameters));
        const metricKeys = this._overlapKeys(items.map(item => item.metrics));

        return (
            <table className={`compare-modal-table ${scrollClass}`}>
                <tbody>
                    {this._renderRow('id', 'ID', 'value idList', items, item => item.id)}
                    {this._renderRow('trialnum', 'Trial No.', 'value', items, item => item.sequenceId.toString())}
                    {this._renderRow('duration', 'Duration', 'value', items, item => item.duration)}
                    {parameterKeys.map(k =>
                        this._renderRow(`space_${k}`, k, 'value', items, item => item.parameters.get(k))
                    )}
                    {metricKeys !== undefined
                        ? metricKeys.map(k =>
                              this._renderRow(`metrics_${k}`, `Metric: ${k}`, 'value', items, item =>
                                  item.metrics.get(k)
                              )
                          )
                        : null}
                </tbody>
            </table>
        );
    }

    render(): React.ReactNode {
        const { onHideDialog, trials, title, showDetails } = this.props;
        const flatten = (m: Map<SingleAxis, any>): Map<string, any> => {
            return new Map(Array.from(m).map(([key, value]) => [key.baseName, value]));
        };
        const inferredSearchSpace = TRIALS.inferredSearchSpace(EXPERIMENT.searchSpaceNew);
        const items: Item[] = trials.map(trial => ({
            id: trial.id,
            sequenceId: trial.sequenceId,
            duration: convertDuration(trial.duration),
            parameters: flatten(trial.parameters(inferredSearchSpace)),
            metrics: flatten(trial.metrics(TRIALS.inferredMetricSpace())),
            intermediates: _parseIntermediates(trial)
        }));

        return (
            <Modal
                isOpen={true}
                containerClassName={contentStyles.container}
                className='compare-modal'
                allowTouchBodyScroll={true}
                dragOptions={dragOptions}
                onDismiss={onHideDialog}
            >
                <div>
                    <div className={contentStyles.header}>
                        <span>{title}</span>
                        <IconButton
                            styles={iconButtonStyles}
                            iconProps={{ iconName: 'Cancel' }}
                            ariaLabel='Close popup modal'
                            onClick={onHideDialog}
                        />
                    </div>
                    <Stack className='compare-modal-intermediate'>
                        {this._intermediates(items)}
                        <Stack className='compare-yAxis'># Intermediate result</Stack>
                    </Stack>
                    {showDetails && <Stack>{this._columns(items)}</Stack>}
                </div>
            </Modal>
        );
    }
}

export default Compare;
