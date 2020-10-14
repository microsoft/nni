import * as React from 'react';
import { DetailsRow, IDetailsRowBaseProps } from '@fluentui/react';
import OpenRow from '../public-child/OpenRow';

interface ExpandableDetailsProps {
    detailsProps: IDetailsRowBaseProps;
    isExpand: boolean;
}

class ExpandableDetails extends React.Component<ExpandableDetailsProps, {}> {
    render(): React.ReactNode {
        const { detailsProps, isExpand } = this.props;
        return (
            <div>
                <DetailsRow {...detailsProps} />
                {isExpand && <OpenRow trialId={detailsProps.item.id} />}
            </div>
        );
    }
}

export default ExpandableDetails;
