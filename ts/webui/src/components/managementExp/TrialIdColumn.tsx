import * as React from 'react';
import { Stack } from '@fluentui/react';
import CopyButton from '../public-child/CopyButton';

interface TrialIdColumnProps {
    port: number;
    id: string;
    status: string;
}

class TrialIdColumn extends React.Component<TrialIdColumnProps, {}> {
    constructor(props: TrialIdColumnProps) {
        super(props);
    }

    render(): React.ReactNode {
        const { port, id, status } = this.props;
        const hostname = window.location.hostname;
        const protocol = window.location.protocol;
        const webuiPortal = `${protocol}//${hostname}:${port}/oview`;
        return (
            <Stack horizontal className='succeed-padding ellipsis idCopy'>
                {status === 'STOPPED' ? (
                    <div className='idColor'>{id}</div>
                ) : (
                    <a
                        href={webuiPortal}
                        className='link toAnotherExp idColor'
                        target='_blank'
                        rel='noopener noreferrer'
                    >
                        {id}
                    </a>
                )}
                <CopyButton value={id} />
            </Stack>
        );
    }
}

export default TrialIdColumn;
