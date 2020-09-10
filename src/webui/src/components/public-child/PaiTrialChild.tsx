import * as React from 'react';
import { DOWNLOAD_IP } from '../../static/const';

interface PaiTrialChildProps {
	logString: string;
	id: string;
	logCollect: boolean;
}

class PaiTrialChild extends React.Component<PaiTrialChildProps, {}> {
	constructor(props: PaiTrialChildProps) {
		super(props);
	}

	render(): React.ReactNode {
		const { logString, id, logCollect } = this.props;
		return (
			<div>
				{logString === '' ? (
					<div />
				) : (
					<div>
						{logCollect ? (
							<a
								target='_blank'
								rel='noopener noreferrer'
								href={`${DOWNLOAD_IP}/trial_${id}.log`}
								style={{ marginRight: 10 }}
							>
								trial stdout
							</a>
						) : (
							<span>trial stdout: {logString}</span>
						)}
					</div>
				)}
			</div>
		);
	}
}

export default PaiTrialChild;
