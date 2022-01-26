import * as React from 'react';
import { Stack } from '@fluentui/react';
import { AllExperimentList } from '../../static/interface';
import CopyButton from '../public-child/CopyButton';

interface TrialIdColumnProps {
    item: AllExperimentList;
}

class TrialIdColumn extends React.Component<TrialIdColumnProps, {}> {
    constructor(props: TrialIdColumnProps) {
        super(props);
    }

    render(): React.ReactNode {
        const { item } = this.props;
        const hostname = window.location.hostname;
        const protocol = window.location.protocol;
        const webuiPortal =
            item.prefixUrl === 'null'
                ? `${protocol}//${hostname}:${item.port}/oview`
                : `${protocol}//${hostname}:${item.port}/${item.prefixUrl}/oview`;
        return (
            <Stack horizontal className='ellipsis idCopy'>
                {item.status === 'STOPPED' ? (
                    <div className='idColor'>{item.id}</div>
                ) : (
                    <a
                        href={webuiPortal}
                        className='link toAnotherExp idColor'
                        target='_blank'
                        rel='noopener noreferrer'
                    >
                        {item.id}
                    </a>
                )}
                <CopyButton value={item.id} />
            </Stack>
        );
    }
}

export default TrialIdColumn;
