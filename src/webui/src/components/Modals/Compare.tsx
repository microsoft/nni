import * as React from 'react';
import { Stack, Modal, IconButton, IDragOptions, ContextualMenu } from 'office-ui-fabric-react';
import ReactEcharts from 'echarts-for-react';
import IntermediateVal from '../public-child/IntermediateVal';
import { TRIALS } from '../../static/datamodel';
import { TableRecord, Intermedia, TooltipForIntermediate } from '../../static/interface';
import { contentStyles, iconButtonStyles } from '../Buttons/ModalTheme';
import '../../static/style/compare.scss';

const dragOptions: IDragOptions = {
	moveMenuItemText: 'Move',
	closeMenuItemText: 'Close',
	menu: ContextualMenu
};

// the modal of trial compare
interface CompareProps {
	compareStacks: Array<TableRecord>;
	cancelFunc: () => void;
}

class Compare extends React.Component<CompareProps, {}> {
	public _isCompareMount!: boolean;
	constructor(props: CompareProps) {
		super(props);
	}

	intermediate = (): React.ReactNode => {
		const { compareStacks } = this.props;
		const trialIntermediate: Array<Intermedia> = [];
		const idsList: string[] = [];
		compareStacks.forEach(element => {
			const trial = TRIALS.getTrial(element.id);
			trialIntermediate.push({
				name: element.id,
				data: trial.description.intermediate,
				type: 'line',
				hyperPara: trial.description.parameters
			});
			idsList.push(element.id);
		});
		// find max intermediate number
		trialIntermediate.sort((a, b) => {
			return b.data.length - a.data.length;
		});
		const legend: string[] = [];
		// max length
		const length = trialIntermediate[0] !== undefined ? trialIntermediate[0].data.length : 0;
		const xAxis: number[] = [];
		trialIntermediate.forEach(element => {
			legend.push(element.name);
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
			legend: {
				// more than 10 trials will hide legend
				data: idsList.length > 10 ? null : idsList
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
			series: trialIntermediate
		};
		return (
			<ReactEcharts
				option={option}
				style={{ width: '100%', height: 418, margin: '0 auto' }}
				notMerge={true} // update now
			/>
		);
	};

	// render table column ---
	initColumn = (): React.ReactNode => {
		const idList: string[] = [];
		const sequenceIdList: number[] = [];
		const durationList: number[] = [];

		const compareStacks = this.props.compareStacks.map(tableRecord => TRIALS.getTrial(tableRecord.id));

		const parameterList: Array<object> = [];
		let parameterKeys: string[] = [];
		if (compareStacks.length !== 0) {
			parameterKeys = Object.keys(compareStacks[0].description.parameters);
		}
		compareStacks.forEach(temp => {
			idList.push(temp.id);
			sequenceIdList.push(temp.sequenceId);
			durationList.push(temp.duration);
			parameterList.push(temp.description.parameters);
		});
		let isComplexSearchSpace;
		if (parameterList.length > 0) {
			isComplexSearchSpace = typeof parameterList[0][parameterKeys[0]] === 'object' ? true : false;
		}
		return (
			<table className='compare-modal-table'>
				<tbody>
					<tr>
						<td className='column'>Id</td>
						{Object.keys(idList).map(key => {
							return (
								<td className='value idList' key={key}>
									{idList[key]}
								</td>
							);
						})}
					</tr>
					<tr>
						<td className='column'>Trial No.</td>
						{Object.keys(sequenceIdList).map(key => {
							return (
								<td className='value idList' key={key}>
									{sequenceIdList[key]}
								</td>
							);
						})}
					</tr>
					<tr>
						<td className='column'>Default metric</td>
						{Object.keys(compareStacks).map(index => {
							const temp = compareStacks[index];
							return (
								<td className='value' key={index}>
									<IntermediateVal trialId={temp.id} />
								</td>
							);
						})}
					</tr>
					<tr>
						<td className='column'>duration</td>
						{Object.keys(durationList).map(index => {
							return (
								<td className='value' key={index}>
									{durationList[index]}
								</td>
							);
						})}
					</tr>
					{isComplexSearchSpace
						? null
						: Object.keys(parameterKeys).map(index => {
								return (
									<tr key={index}>
										<td className='column' key={index}>
											{parameterKeys[index]}
										</td>
										{Object.keys(parameterList).map(key => {
											return (
												<td key={key} className='value'>
													{parameterList[key][parameterKeys[index]]}
												</td>
											);
										})}
									</tr>
								);
						  })}
				</tbody>
			</table>
		);
	};

	componentDidMount(): void {
		this._isCompareMount = true;
	}

	componentWillUnmount(): void {
		this._isCompareMount = false;
	}

	render(): React.ReactNode {
		const { cancelFunc } = this.props;

		return (
			<Modal
				isOpen={true}
				containerClassName={contentStyles.container}
				className='compare-modal'
				allowTouchBodyScroll={true}
				dragOptions={dragOptions}
			>
				<div>
					<div className={contentStyles.header}>
						<span>Compare trials</span>
						<IconButton
							styles={iconButtonStyles}
							iconProps={{ iconName: 'Cancel' }}
							ariaLabel='Close popup modal'
							onClick={cancelFunc}
						/>
					</div>
					<Stack className='compare-modal-intermediate'>
						{this.intermediate()}
						<Stack className='compare-yAxis'># Intermediate result</Stack>
					</Stack>
					<Stack>{this.initColumn()}</Stack>
				</div>
			</Modal>
		);
	}
}

export default Compare;
