import * as React from 'react';
import * as copy from 'copy-to-clipboard';
import { Stack, PrimaryButton, Pivot, PivotItem } from 'office-ui-fabric-react';
import { Trial } from '../../static/model/trial';
import { EXPERIMENT, TRIALS } from '../../static/datamodel';
import JSONTree from 'react-json-tree';
import PaiTrialLog from '../public-child/PaiTrialLog';
import TrialLog from '../public-child/TrialLog';
import MessageInfo from '../Modals/MessageInfo';
import '../../static/style/overview.scss';
import '../../static/style/copyParameter.scss';

interface OpenRowProps {
	trialId: string;
}

interface OpenRowState {
	typeInfo: string;
	info: string;
	isHidenInfo: boolean;
}

class OpenRow extends React.Component<OpenRowProps, OpenRowState> {
	constructor(props: OpenRowProps) {
		super(props);
		this.state = {
			typeInfo: '',
			info: '',
			isHidenInfo: true
		};
	}

	hideMessageInfo = (): void => {
		this.setState(() => ({ isHidenInfo: true }));
	};

	/**
	 * info: message content
	 * typeInfo: message type: success | error...
	 * continuousTime: show time, 2000ms
	 */
	getCopyStatus = (info: string, typeInfo: string): void => {
		this.setState(() => ({ info, typeInfo, isHidenInfo: false }));
		setTimeout(this.hideMessageInfo, 2000);
	};

	copyParams = (trial: Trial): void => {
		// get copy parameters
		const params = JSON.stringify(trial.description.parameters, null, 4);
		if (copy.default(params)) {
			this.getCopyStatus('Success copy parameters to clipboard in form of python dict !', 'success');
		} else {
			this.getCopyStatus('Failed !', 'error');
		}
	};

	render(): React.ReactNode {
		const { isHidenInfo, typeInfo, info } = this.state;
		const trialId = this.props.trialId;
		const trial = TRIALS.getTrial(trialId);
		const logPathRow = trial.info.logPath || "This trial's log path is not available.";
		return (
			<Stack className='openRow'>
				<Stack className='openRowContent'>
					<Pivot>
						<PivotItem headerText='Parameters' key='1' itemIcon='TestParameter'>
							{trial.info.hyperParameters !== undefined ? (
								<Stack id='description'>
									<Stack className='bgHyper'>
										<JSONTree
											hideRoot={true}
											shouldExpandNode={(): boolean => true} // default expandNode
											getItemString={(): null => null} // remove the {} items
											data={trial.description.parameters}
										/>
									</Stack>
									<Stack horizontal className='copy'>
										<PrimaryButton
											onClick={this.copyParams.bind(this, trial)}
											text='Copy as json'
											styles={{ root: { width: 128, marginRight: 10 } }}
										/>
										{/* copy success | failed message info */}
										{!isHidenInfo && <MessageInfo typeInfo={typeInfo} info={info} />}
									</Stack>
								</Stack>
							) : (
								<Stack className='logpath'>
									<span className='logName'>Error: </span>
									<span className='error'>{`This trial's parameters are not available.'`}</span>
								</Stack>
							)}
						</PivotItem>
						<PivotItem headerText='Log' key='2' itemIcon='M365InvoicingLogo'>
							{// FIXME: this should not be handled in web UI side
							EXPERIMENT.trainingServicePlatform !== 'local' ? (
								<PaiTrialLog
									logStr={logPathRow}
									id={trialId}
									logCollection={EXPERIMENT.logCollectionEnabled}
								/>
							) : (
								<TrialLog logStr={logPathRow} id={trialId} />
							)}
						</PivotItem>
					</Pivot>
				</Stack>
			</Stack>
		);
	}
}

export default OpenRow;
